import mcts
import random
import pyximport

pyximport.install(setup_args={"script_args": ['--cython-cplus']})
from engine import *

def visualize_tree(root, output_file="tree.dot"):
    from graphviz import Digraph

    def add_nodes_edges(dot, node, parent_id=None, edge_label=""):
        node_id = str(id(node))
        if node.num_sims == 0:
            node_id = parent_id + '_null'

        value = 'x' if node.num_sims == 0 else node.total_value / node.num_sims
        dot.node(node_id, label=f"Value: {value}\nSims: {node.num_sims}\n\n{node.game.display()}")

        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        for move, child_set in zip(node.moves, node.children):
            for child in child_set.nodes:
                add_nodes_edges(dot, child, node_id, move)

    dot = Digraph()
    add_nodes_edges(dot, root)
    dot.render(output_file, format='pdf')


class F:
    def __init__(self, x):
        self.x = x

    def item(self):
        return self.x

def model(games):
    return (None, F(random.random()))

if __name__ == '__main__':
    g = Game()
    tree = mcts.Node(g)
    for i in range(50):
        mcts.mcts_step(model, tree, 1)
        visualize_tree(tree, f'tree_{i}')
