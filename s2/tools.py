from deap import gp
import pygraphviz as pgv

def draw(individual, path):
    nodes, edges, labels = gp.graph(individual)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(path)