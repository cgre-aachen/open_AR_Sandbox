import panel as pn
pn.extension()

from bokeh.plotting import figure


def test_panel():
    p1 = figure(width=300, height=300, name='Scatter')
    p1.scatter([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])

    p2 = figure(width=300, height=300, name='Line')
    p2.line([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])

    tabs = pn.Tabs(('Scatter', p1), p2)
    tabs.show()

test_panel()
