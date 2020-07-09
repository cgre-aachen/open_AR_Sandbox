import sandbox as sb
import matplotlib.pyplot as plt
sensor = sb.Sensor(name='dummy')

def test_init():
    module = sb.projector.CmapModule(extent=sensor.extent)
    print(module)

def test_render_frame():
    module = sb.projector.CmapModule(extent=sensor.extent)
    frame = sensor.get_frame()
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    fig.show()

def test_change_cmap():
    module = sb.projector.CmapModule(extent=sensor.extent)
    frame = sensor.get_frame()
    fig1, ax1 = plt.subplots()
    module.render_frame(frame, ax1)
    fig1.show()
    cmap = plt.cm.get_cmap('Accent_r')
    fig2, ax2 = plt.subplots()
    module.set_cmap(cmap)
    module.render_frame(frame, ax2)
    fig2.show()

#%%
frame = sensor.get_frame()
fig, ax = plt.subplots()
col = ax.pcolormesh(frame)
fig.show()

#%%
for i in range (10):
    col.set_array(sensor.get_frame().ravel())
    fig.show()
#%%

#%%
fig.show()
col.remove()
fig.show()

col.up
#%%
fig.canvas.draw()
fig.canvas.flush_events()
#%%
fig.show()
#%%
cmap = plt.cm.get_cmap("hot")
col.set_cmap(cmap)

#%%
fig.show()
#%%
plt.
