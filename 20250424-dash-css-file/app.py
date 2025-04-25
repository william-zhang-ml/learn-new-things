"""
Static webpage with style defined in a separate CSS file.
"""
from dash import Dash, html


app = Dash()
app.layout = html.H1('Hello world', id='page-header')


if __name__ == '__main__':
    app.run()
