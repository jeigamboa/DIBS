import dash
from dash import html, dcc, Output, Input, State
import dash_leaflet as dl
import plotly.express as px
import plotly.graph_objs as go

app = dash.Dash()
app.layout = html.Div([
    dl.Map([dl.TileLayer(), dl.LayerGroup(id="container", children=[])], id="map",
           center=(56, 10), zoom=10, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
])

@app.callback(Output("container", "children"), [Input("map", "click_lat_lng")], [State("container", "children")])
def add_marker(click_lat_lng, children):
    children.append(dl.Marker(position=click_lat_lng))
    return children

if __name__ == '__main__':
    app.run_server()