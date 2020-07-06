#%%
import sandbox as sb
import matplotlib.pyplot as plt
#%%
sensor = sb.Sensor(name='dummy')


def test_init():
    module = sb.SimpleTopoModule(extent=sensor.extent)
    print(module)
    
def test_update():
    module = sb.SimpleTopoModule(extent=sensor.extent)
    fig, ax = plt.subplots()
    module.update(sensor.get_frame(), ax)
    fig.show()

def test_thread_update():
    projector = sb.Projector(use_panel=True)
    thread = sb.MainThread(sensor, projector)

    module = sb.SimpleTopoModule(extent=sensor.extent)
    thread.modules = [module]

    thread.update()

