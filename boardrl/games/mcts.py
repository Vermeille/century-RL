import random
import numpy as np


def game_step(game, action, opponent_fn):
    player = game.current_player()
    score = game.diff_points_for(player)
    game.play_str(action)
    while game.current_player() != player:
        if game.ended():
            break
        opponent_fn(game)
    return game.diff_points_for(player) - score


def simulate_to_end(game, max_steps):
    player = game.current_player()
    step = 0
    last_score = game.diff_points_for(player)
    rewards = []
    while True:
        if game.ended():
            break

        if step >= max_steps:
            break

        game.play_str(random.choice(game.moves))

        if game.current_player() == player:
            points = game.diff_points_for(player)
            rewards.append(points - last_score)
            last_score = points

        step += 1
    return rewards


def discount(rewards, gamma):
    return sum(gamma**i * r for i, r in enumerate(rewards))


class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def draw(self):
        out = "digraph G {\n"
        out += self._draw()
        out += "}\n"
        return out

    def _draw(self):
        out = f'"{self}" [label="{self.total_reward / self.visits}"];\n'
        for child in self.children:
            out += f'"{self}" -> "{child}" [label="{child.action}"];\n'
            out += child._draw()
        return out

    def __str__(self):
        from textwrap import indent

        return (
            f"{self.action} ({self.total_reward / self.visits if self.visits else None}, {self.visits})"
            + ("\n" if len(self.children) else "")
            + indent("\n".join(str(c) for c in self.children), "  ")
        )


class MCTS:
    def __init__(self, me, discount_factor, max_unroll):
        self.me = me
        self.discount_factor = discount_factor
        self.root_node = Node()
        self.max_unroll = max_unroll

    def _select(self, game):
        """Selection phase using UCT"""
        path = []
        current = self.root_node

        while True:
            path.append(current)

            # Check terminal state
            if game.ended():
                return path

            # Check expandable
            unexplored = [
                a
                for a in game.moves
                if not any(c.action == a for c in current.children) and a in game.moves
            ]
            if unexplored:
                return path

            # Select best child using UCT
            current = self._select_child(current, game.moves)
            reward = game_step(
                game, current.action, lambda g: g.play_str(random.choice(g.moves))
            )
            self._backpropagate(current, reward)

    def _select_child(self, node, moves):
        """UCT selection with exploration/exploitation tradeoff"""
        log_n = np.log(node.visits + 1e-10)

        def uct(child):
            if child.visits == 0:
                return float("inf")
            return (child.total_reward / child.visits) + np.sqrt(
                2 * log_n / child.visits
            )

        return max([c for c in node.children if c.action in moves], key=uct)

    def _expand(self, path, game):
        """Expansion phase - add one child node"""
        node = path[-1]
        unexplored = [
            a for a in game.moves if not any(c.action == a for c in node.children)
        ]

        if not unexplored:
            return None

        # Choose first unexplored action
        action = unexplored[0]
        new_node = Node(parent=node, action=action)
        node.children.append(new_node)
        return new_node

    def _backpropagate(self, node, reward):
        """Update statistics along the path"""
        while node:
            node.visits += 1
            node.total_reward += reward
            reward = self.discount_factor * reward
            node = node.parent

    def search(self, game, iterations=1000):
        """Run MCTS and return best action"""
        assert game.current_player() == self.me, f"Current player is not {self.me}"
        for _ in range(iterations):
            g = game.copy()
            path = self._select(g)

            if g.ended():
                reward = g.diff_points_for(self.me)
            else:
                new_node = self._expand(path, g)
                if new_node:
                    path.append(new_node)
                    game_step(
                        g,
                        new_node.action,
                        lambda g: g.play_str(random.choice(g.moves)),
                    )
                if g.ended():
                    reward = g.diff_points_for(self.me)
                else:
                    assert g.current_player() == self.me
                    rewards = simulate_to_end(g, self.max_unroll)
                    reward = discount(rewards, self.discount_factor)

            self._backpropagate(path[-1], reward)

        return [c.visits for c in self.root_node.children]
