from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
import random

def make_document(doc):
    source = ColumnDataSource({'x': [], 'y': [], 'color': []})

    def update():
        new = {'x': [random.random()], 'y': [random.random()], 'color': [random.choice(['red', 'blue', 'green'])]}
        source.stream(new)

    doc.add_periodic_callback(update, 1000)

    fig = figure(title = 'Streaming Circle Plot!', x_range = [0, 1], y_range = [0, 1])
    fig.circle(source = source, x = 'x', y = 'y', color = 'color', size = 10)

    doc.title = "Now with live updating!"
    doc.add_root(fig)

apps = {'/': Application(FunctionHandler(make_document))}

io_loop = IOLoop.current()
server = Server(applications = {'/': Application(FunctionHandler(make_document))}, io_loop = io_loop, port = 0)
server.start()
server.show('/')
#io_loop.start()