#%%
import sandbox as sb
import matplotlib.pyplot as plt
sensor = sb.Sensor(name='dummy')


def test_init():
    module = sb.ContourLinesModule(extent=sensor.extent)
    print(module)


def test_update():
    module = sb.ContourLinesModule(extent=sensor.extent)
    fig, ax = plt.subplots()
    module.plot_contour_lines(sensor.get_frame(), ax)
    fig.show()


