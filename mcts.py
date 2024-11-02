import random
import math
import torch

def uct(parent, child):
    value = 0.5
    if child.num_sims > 0:
        value = 1 - child.total_value / child.num_sims

    prior = 1/len(parent.children) * math.log(parent.num_sims) / (child.num_sims + 1)
    return value + prior


class NodesSet:
    def __init__(self, parent=None):
        self.nodes = []
        self.parent = parent
        # those values summarize the children's
        self.total_value = 0
        self.num_sims = 0

    def select(self):
        return max(self.nodes, key=lambda n: uct(self.parent, n)).select()

    def backup(self, value):
        self.total_value += value
        self.num_sims += 1

        if self.parent is not None:
            # back to opponent, change the value to their view
            self.parent.backup(1 - value)


class Node:
    def __init__(self, game, parent=None):
        self.game = game
        self.parent = parent
        self.total_value = 0
        self.num_sims = 0
        self.children = []
        self.moves = []
        
    def simulate(self, nn):
        if self.game.ended():
            if (self.game.p0.points() > self.game.p1.points()) == (self.game.state == 0): # State.P0_TURN
                return 1
            else:
                return -1
        with torch.no_grad():
            return nn([self.game.display()])[1].item()

    def expand(self, num_samples):
        if len(self.children) > 0 or self.game.ended():
            return


        self.moves = self.game.gen_move()
        for move in self.moves:
            node_set = NodesSet(parent=self)
            for _ in range(num_samples):
                g = self.game.copy()
                g.play_str(move)
                node_set.nodes.append(Node(g, parent=node_set))
            self.children.append(node_set)

    def backup(self, value):
        self.total_value += value
        self.num_sims += 1

        if self.parent is not None:
            # send to the node set
            self.parent.backup(value)

    def select(self):
        if len(self.children) == 0:
            return self
        return max(self.children, key=lambda n: uct(self, n)).select()


    def moves_distribution(self):
        assert len(self.moves) > 0
        return self.moves, [c.num_sims for c in self.children]


def mcts_step(model, root, num_samples):
    node = root.select()
    node.expand(num_samples)
    if len(node.children) > 0:
        node = random.choice(random.choice(node.children).nodes)
    value = node.simulate(model)
    node.backup(value)

def MCTS(model, game, root, budget, num_samples):
    if root is None:
        root = Node(game)
    else:
        try:
            idx = ([n.game.display() for n in root.nodes]).index(game.display())
            root = root.nodes[idx]
            print('reusing root', root.num_sims)
        except ValueError:
            print('need new tree')
            root = Node(game)

    while len(root.children) == 0 or len(root.children) * budget > root.num_sims:
        mcts_step(model, root, num_samples)
    moves, moves_count = root.moves_distribution()
    moves_probs = (torch.tensor(moves_count) / sum(moves_count)) ** (1 / 0.7)
    move_idx = torch.multinomial(moves_probs, 1)
    new_root = root.children[move_idx]
    new_root.parent = None
    print(moves_count)
    return moves[move_idx], new_root, list(zip(moves, moves_count))
