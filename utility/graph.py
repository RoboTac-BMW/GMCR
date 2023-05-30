import igraph as ig
import numpy as np
import plotly.graph_objs as go
import networkx as nx
import sys
LIB_PMC_PATH = '/home/mgentner/libs/pmc'
sys.path.append(LIB_PMC_PATH)
from pmc import pmc


class Graph:

    def __init__(self, edges, n_nodes):

        self.edges = edges
        self.n = n_nodes
        self.max_clique = None

    def __str__(self):

        return "Graph with {} nodes".format(self.n)

    def cliques(self):

        max_clique = pmc(self.edges[:, 1], self.edges[:, 0], self.n, self.edges.shape[0])

        return [max_clique]

    def degeneracy(self):
        g = nx.Graph()
        g.add_edges_from(self.edges)
        g.remove_edges_from(nx.selfloop_edges(g))
        return max(nx.core_number(g).values())

    def display(self, inliers=None):

        # create a layout of the graph with igraph
        graph = ig.Graph(self.edges, directed=False)
        graph_layout = graph.layout('kk', dim=3)
        graph_layout = np.array(graph_layout)

        # Convert edges to edge start- and end-positions
        edge_x = []
        edge_y = []
        edge_z = []
        for i in range(self.edges.shape[0]):

            edge_x += [graph_layout[self.edges[i, 0], 0], graph_layout[self.edges[i, 1], 0], None]
            edge_y += [graph_layout[self.edges[i, 0], 1], graph_layout[self.edges[i, 1], 1], None]
            edge_z += [graph_layout[self.edges[i, 0], 2], graph_layout[self.edges[i, 1], 2], None]
        # Create labels for nodes
        text_labels = [str(i) for i in range(graph_layout.shape[0])]

        outliers = (~inliers.astype(bool))
        inliers = inliers.astype(bool)
        print("Num outliers: {}".format(outliers.sum()))
        print("Num inliers: {}".format(inliers.sum()))

        trace1 = go.Scatter3d(x=edge_x,
                              y=edge_y,
                              z=edge_z,
                              mode='lines',
                              line=dict(color='rgb(125,125,125)', width=2),
                              hoverinfo='none'
                              )

        trace2 = go.Scatter3d(x=graph_layout[inliers, 0],
                              y=graph_layout[inliers, 1],
                              z=graph_layout[inliers, 2],
                              mode='markers',
                              name='nodes',
                              marker=dict(symbol='circle',
                                          size=6,
                                          #color=group,
                                          #colorscale='Viridis',
                                          line=dict(color='green', width=0.5)
                                          ),
                              text=text_labels,
                              hoverinfo='text'
                              )

        trace3 = go.Scatter3d(x=graph_layout[outliers, 0],
                              y=graph_layout[outliers, 1],
                              z=graph_layout[outliers, 2],
                              mode='markers',
                              name='nodes',
                              marker=dict(symbol='circle',
                                          size=6,
                                          #color=group,
                                          #colorscale='Viridis',
                                          line=dict(color='red', width=0.5)
                                          ),
                              text=text_labels,
                              hoverinfo='text'
                              )



        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
             title="Simple Random Undirected Graph",
             width=1000,
             height=1000,
             showlegend=False,
             scene=dict(
                 xaxis=dict(axis),
                 yaxis=dict(axis),
                 zaxis=dict(axis),
            ),
         margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
               dict(
               showarrow=False,
                text="No Text to put here",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ])

        data = [trace1, trace3, trace2]

        fig = go.Figure(data=data, layout=layout)
        fig.show()


def no_common_edge(edges_a, edges_b):
    set_a = set(tuple(map(tuple, edges_a)))
    set_b = set(tuple(map(tuple, edges_b)))

    return len(set_a.intersection(set_b)) == 0


def edge_set_equivalent(edges_a, edges_b):
    set_a = set(tuple(map(tuple, edges_a)))
    set_b = set(tuple(map(tuple, edges_b)))

    return set_a == set_b
