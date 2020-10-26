from sandbox import _test_data
from sandbox.modules import SeismicModule
import matplotlib.pyplot as plt
import pytest
import numpy as np
import pandas as pd

df = pd.read_pickle(_test_data['test']+"arucos.pkl")
file = np.load(_test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
frame = frame + np.abs(np.amin(frame))
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': df,
                    'cmap': plt.cm.get_cmap('viridis'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}


def test_init():
    seis = SeismicModule()

def test_scale_frame():
    seis = SeismicModule()
    new_frame = seis.scale_linear(frame, 5, 2)
    assert np.amin(new_frame) == 2 and np.amax(new_frame) == 5

def test_smooth_topo():
    seis = SeismicModule()
    new_frame = seis.smooth_topo(frame, 10, 5)
    print(new_frame==frame)
    plt.imshow(frame, cmap="viridis", origin="lower")
    plt.show()

    plt.imshow(new_frame, cmap="viridis",  origin="lower")
    plt.show()

def test_create_velocity_model():
    seis = SeismicModule()
    plt.imshow(frame.T, cmap="viridis")
    plt.show()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=True)

    seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=True)

def test_create_time_axis():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    time_range = seis.create_time_axis(t0=0, tn=1000)
    print(time_range)

def test_create_source_wavelet():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=True, show_model=True)
    seis.create_source(name ="src1", f0=0.025, source_coordinates=(700,200), show_wavelet=False, show_model=True)

def test_create_time_function():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.create_time_function()

def test_solve_PDE():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.create_time_function()
    seis.solve_PDE()

def test_inject_source():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    src = seis.create_source(name="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    src0 = seis.create_source(name="src0", f0=0.025, source_coordinates=(300,200), show_wavelet=False, show_model=False)
    src1 = seis.create_source(name="src1", f0=0.025, source_coordinates=(300,1500), show_wavelet=False, show_model=False)

    seis.inject_source(src)
    seis.inject_source(src0)
    seis.inject_source(src1)

    seis.show_velocity(seis.model, source=seis.src, receiver=seis.rec)

def test_operator():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    src = seis.create_source(name="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()

def test_receivers():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    rec = seis.create_receivers(name='rec', n_receivers=100, depth_receivers=200, show_receivers=True )

def test_interpolate_receivers():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    rec=seis.create_receivers(name='rec', n_receivers=100, depth_receivers=200, show_receivers=True)
    seis.interpolate_receiver(rec)

def test_operator_receiver():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    rec=seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver(rec)

    src = seis.create_source(name="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()

    seis.show_velocity(seis.model, source=src.coordinates.data,
                       receiver=rec.coordinates.data[::4, :])

def test_show_shot_record():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    #seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=True)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    rec = seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver(rec)

    src = seis.create_source(name="src", f0=0.01, source_coordinates=(500,20), show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()
    seis.show_shotrecord(rec.data, seis.model, 0, 1000)

def test_show_wavefield():
    seis = SeismicModule()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, nbl=40, show_velocity=False)
    #seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    rec=seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver(rec)

    src = seis.create_source(name="src", f0=0.025, source_coordinates=(500, 400), show_wavelet=False, show_model=False)
    src1 = seis.create_source(name="src1", f0=0.025, source_coordinates=(800, 800), show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.inject_source(src1)
    seis.operator_and_solve()

    seis.show_velocity(seis.model, source=seis.src_coordinates,
                       receiver=rec.coordinates.data)
    seis.show_wavefield(timeslice=10)
    seis.show_wavefield(timeslice=50)
    seis.show_wavefield(timeslice=100)
    seis.show_wavefield(timeslice=200)
    seis.show_wavefield(timeslice=300)
    seis.show_wavefield(timeslice=400)

    seis.show_wavefield(timeslice=5000)

def test_init_all_velocity_model():
    seis = SeismicModule()
    file = np.load(_test_data['test'] + "frame1.npz")
    frame = seis.crop_frame(origin=(20,20), width=200, height=180, frame=file['arr_0'])
    seis.init_model(vmin=2, vmax=4, frame=np.transpose(frame))
    plt.imshow(frame, origin="lower", cmap ="gist_earth")
    plt.show()
    seis.show_velocity(seis.model)

def test_insert_aruco_source():
    seis = SeismicModule()
    file = np.load(_test_data['test'] + "frame1.npz")
    frame = seis.crop_frame(origin=(10, 10), width=230, height=180, frame=file['arr_0'])
    seis.init_model(vmin=2, vmax=4, frame=np.transpose(frame))

    marker = pytest.sb_params['marker']
    seis.xy_aruco=marker.loc[marker.is_inside_box, ('box_x', 'box_y')].values
    seis.insert_aruco_source()

    seis.show_velocity(seis.model, source=seis.src_coordinates)
    print(seis.src_coordinates)

def test_run_aruco_source():
    seis = SeismicModule()
    file = np.load(_test_data['test'] + "frame1.npz")
    frame = seis.crop_frame(origin=(10, 10), width=230, height=180, frame=file['arr_0'])
    seis.init_model(vmin=2, vmax=4, frame=np.transpose(frame), nbl=40)

    marker = pytest.sb_params['marker']
    seis.xy_aruco=marker.loc[marker.is_inside_box, ('box_x', 'box_y')].values
    seis.insert_aruco_source()

    seis.show_velocity(seis.model, source=seis.src_coordinates)
    print(seis.src_coordinates)
    seis.operator_and_solve()

    seis.show_velocity(seis.model, source=seis.src_coordinates)
    seis.show_wavefield(timeslice=10)
    seis.show_wavefield(timeslice=50)
    seis.show_wavefield(timeslice=100)
    seis.show_wavefield(timeslice=200)
    seis.show_wavefield(timeslice=300)
    seis.show_wavefield(timeslice=400)
    seis.show_wavefield(timeslice=5000)

def test_panel_plotting():
    seis = SeismicModule()
    marker = pytest.sb_params['marker']
    seis.xy_aruco = marker.loc[marker.is_inside_box, ('box_x', 'box_y')].values
    file = np.load(_test_data['test'] + "frame1.npz")
    seis.frame = seis.crop_frame(origin=(30, 30), width=230, height=180, frame=file['arr_0'])
    seis.run_simulation()
    seis.timeslice = 50
    sb_params = seis.update(pytest.sb_params)
    fig = sb_params["fig"]
    fig.show()