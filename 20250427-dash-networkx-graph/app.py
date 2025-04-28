"""
Showing a NetworkX graph in a Dash dashboard.
1. Define the graph
2. Format node data for marker scatter
3. Format node data for line scatter
4. Define the dashboard: div -> graph -> marker scatter, line scatter
"""
import dash
from dash import html, dcc
import plotly.graph_objects as go
import networkx as nx


# define graph
graph = nx.DiGraph()
graph.add_nodes_from([0, 1, 10, 11])
graph.add_edges_from([[0, 1], [10, 11], [0, 11]])
pos = {0: [0, 0], 1: [1, 0], 10: [0, 1], 11: [1, 1]}

# format nodes data for display
node_x = []
node_y = []
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

# format edge data for display
edge_x = []
edge_y = []
for src, dest in graph.edges():
    x0, y0 = pos[src]
    x1, y1 = pos[dest]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# create graph app
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        marker={'size': 50, 'color': 'steelblue'},
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line={'width': 2, 'color': 'dimgray'},
                        hoverinfo='none'
                    )
                ],
                layout=go.Layout(
                    title='NetworkX Graph in Dash',
                    showlegend=False,
                    hovermode='closest'
                )
            ),
            style={'height': '100%', 'width': '100%'}
        )
    ],
    style={
        'height': '90vh',
        'width': '100%',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'border': '1px solid black'
    }
)


if __name__ == '__main__':
    app.run_server(debug=True)
