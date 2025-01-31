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


class Simulate:
    def __init__(self, max_steps, discount_factor, num_unrolls=1):
        self.max_steps = max_steps
        self.discount_factor = discount_factor
        self.num_unrolls = num_unrolls

    def simulate(self, game):
        player = game.current_player()
        step = 0
        rewards = []
        while True:
            if game.ended():
                break

            if step >= self.max_steps:
                break

            assert game.current_player() == player
            rewards.append(
                game_step(
                    game,
                    random.choice(game.moves),
                    lambda g: g.play_str(random.choice(g.moves)),
                )
            )
            step += 1
        return discount(rewards, self.discount_factor)

    async def __call__(self, game):
        rewards = sum(self.simulate(game.copy()) for _ in range(self.num_unrolls))
        return rewards / self.num_unrolls


def discount(rewards, gamma):
    return sum(gamma**i * r for i, r in enumerate(rewards))


class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.availability = 1

    def draw(self):
        out = "digraph G {\n"
        out += self._draw()
        out += "}\n"
        return out

    def _draw(self):
        if self.parent:
            ratio = self.visits / self.parent.visits
        else:
            ratio = 1  # Root node

        # Interpolate color based on ratio
        blue = (128, 128, 255)
        red = (255, 128, 128)
        color = (
            int(blue[0] + (red[0] - blue[0]) * ratio),
            int(blue[1] + (red[1] - blue[1]) * ratio),
            int(blue[2] + (red[2] - blue[2]) * ratio),
        )
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        out = f'"{id(self)}" [label="{self.total_reward / self.visits if self.visits else float("nan"):.3f} {self.visits}", style=filled, fillcolor="{color_hex}"];\n'
        for child in self.children:
            if child.visits == 0:
                continue
            out += f'"{id(self)}" -> "{id(child)}" [label="{child.action}"];\n'
            out += child._draw()
        return out

    def __str__(self):
        from textwrap import indent

        return (
            f"{self.action} ({self.total_reward / self.visits if self.visits else "NaN"}, {self.visits})"
            + ("\n" if len(self.children) else "")
            + indent("\n".join(str(c) for c in self.children), "  ")
        )


class MCTS:
    def __init__(self, me, discount_factor, eval_fn):
        self.me = me
        self.discount_factor = discount_factor
        self.root_node = Node()
        self.eval_fn = eval_fn

    def _select(self, game):
        """Selection phase using UCT"""
        path = []
        rewards = [0]
        current = self.root_node

        while True:
            path.append(current)

            # Check expandable
            self._expand(current, game.moves)

            # Check terminal state
            if game.ended() or current.visits == 0:
                return path, rewards

            # Select best child using UCT
            current = self._select_child(current, game.moves)
            reward = game_step(
                game, current.action, lambda g: g.play_str(random.choice(g.moves))
            )
            rewards.append(reward)

    def _select_child(self, node, moves):
        """UCT selection with exploration/exploitation tradeoff"""
        log_n = np.log(node.visits + 1e-10)

        def ucb(child):
            if child.visits == 0:
                return float("inf")
            return (child.total_reward / child.visits) + 0.01 * np.sqrt(
                log_n / child.visits
            )

        def uct(child):
            prob_action = child.availability / node.visits
            if child.visits == 0:
                if random.random() < prob_action:
                    return float("inf")
                else:
                    return float("-inf")
            return (child.total_reward / child.visits) + 0.1 * prob_action * np.sqrt(
                node.visits
            ) / (1 + child.visits)

        return max([c for c in node.children if c.action in moves], key=uct)

    def _expand(self, parent, moves):
        """Expansion phase - add one child node"""
        # Choose first unexplored action
        for action in moves:
            node = [c for c in parent.children if c.action == action]
            if node:
                assert len(node) == 1
                node[0].availability += 1
            else:
                child = Node(parent=parent, action=action)
                parent.children.append(child)

    def _backpropagate(self, node, reward):
        """Update statistics along the path"""
        assert len(node) == len(reward)
        total_reward = 0
        for node, reward in zip(reversed(node), reversed(reward)):
            total_reward = self.discount_factor * total_reward + reward
            node.visits += 1
            node.total_reward += total_reward

    async def search(self, game, iterations=1000):
        """Run MCTS and return best action"""
        import time
        from subprocess import Popen

        self.root_node.visits = 1
        self.root_node.total_reward = await self.eval_fn(game)

        assert game.current_player() == self.me, f"Current player is not {self.me}"
        for _ in range(iterations):
            g = game.copy()
            path, rewards = self._select(g)

            if not g.ended():
                assert (
                    game.current_player() == self.me
                ), f"Current player is not {self.me}"

                if not g.ended():
                    assert g.current_player() == self.me
                    value = await self.eval_fn(g)
                    rewards[-1] += value

            self._backpropagate(path, rewards)

        return [c.visits for c in self.root_node.children]
