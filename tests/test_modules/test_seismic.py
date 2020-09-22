from sandbox import _package_dir, _test_data
from sandbox.modules import SeismicModule
import matplotlib.pyplot as plt
import pytest
import numpy as np
file = np.load(_test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
frame = frame + np.abs(np.amin(frame))
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

devito_dir = _package_dir+'/../../devito/'

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': [],
                    'cmap': plt.cm.get_cmap('viridis'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}

def test_import_libraries():
    import sys, os
    sys.path.append(os.path.abspath(devito_dir)+'/examples/seismic/')
    from model import Model
    from plotting import plot_velocity
    from source import RickerSource
    from devito import TimeFunction, Eq, solve, Operator

def test_init():
    seis = SeismicModule(devito_dir=devito_dir)

def test_scale_frame():
    seis = SeismicModule(devito_dir=devito_dir)
    new_frame = seis.scale_linear(frame, 5, 2)
    assert np.amin(new_frame) == 2 and np.amax(new_frame) == 5

def test_smooth_topo():
    seis = SeismicModule(devito_dir=devito_dir)
    new_frame = seis.smooth_topo(frame, 10, 5)
    print(new_frame==frame)
    plt.imshow(frame, cmap="viridis", origin="lower left")
    plt.show()

    plt.imshow(new_frame, cmap="viridis",  origin="lower left")
    plt.show()

def test_create_velocity_model():
    seis = SeismicModule(devito_dir=devito_dir)
    plt.imshow(frame.T, cmap="viridis")
    plt.show()
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=True)

    seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=True)

def test_create_time_axis():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    time_range = seis.create_time_axis(t0=0, tn=1000)
    print(time_range)

def test_create_source_wavelet():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=True, show_model=True)
    seis.create_source(name ="src1", f0=0.025, source_coordinates=(700,200), show_wavelet=False, show_model=True)

def test_create_time_function():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.create_time_function()

def test_solve_PDE():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_source(name ="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.create_time_function()
    seis.solve_PDE()

def test_inject_source():
    seis = SeismicModule(devito_dir=devito_dir)
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

def test_operator():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    src = seis.create_source(name="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()

def test_receivers():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.create_receivers(name='rec', n_receivers=100, depth_receivers=200, show_receivers=True )

def test_interpolate_receivers():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.create_receivers(name='rec', n_receivers=100, depth_receivers=200, show_receivers=True)
    seis.interpolate_receiver()

def test_operator_receiver():
    seis = SeismicModule(devito_dir=devito_dir)
    seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver()

    src = seis.create_source(name="src", f0=0.025, source_coordinates=None, show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()

    seis.plot_velocity(seis.model, source=seis.src.coordinates.data,
              receiver=seis.rec.coordinates.data[::4, :])

def test_plot_shot_record():
    seis = SeismicModule(devito_dir=devito_dir)
    #seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=True)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver()

    src = seis.create_source(name="src", f0=0.01, source_coordinates=(500,20), show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()
    seis.plot_shotrecord(seis.rec.data, seis.model, 0, 1000)

def test_plot_wavefield():
    seis = SeismicModule(devito_dir=devito_dir)
    # seis.create_velocity_model(frame, vmax=5, vmin=2, sigma_x=5, sigma_y=5, show_velocity=False)
    seis.create_velocity_model(None, norm=False, smooth=False, show_velocity=False)
    seis.create_time_axis(t0=0, tn=1000)
    seis.create_time_function()
    seis.solve_PDE()

    seis.create_receivers(name='rec', n_receivers=100, depth_receivers=20, show_receivers=False)
    seis.interpolate_receiver()

    src = seis.create_source(name="src", f0=0.01, source_coordinates=(500, 20), show_wavelet=False, show_model=False)
    seis.inject_source(src)
    seis.operator_and_solve()

    seis.plot_velocity(seis.model, source=seis.src.coordinates.data,
                       receiver=seis.rec.coordinates.data)

    extent = [seis.model.origin[0], seis.model.origin[0] + 1e-3 * seis.model.shape[0] * seis.model.spacing[0],
              seis.model.origin[1] + 1e-3 * seis.model.shape[1] * seis.model.spacing[1], seis.model.origin[1]]
    data_param = dict(vmin=-1e0, vmax=1e0, cmap=plt.get_cmap('Greys'), aspect=1, interpolation='none', extent=extent)
    model_param = dict(vmin=1.5, vmax=2.5, cmap=plt.get_cmap('GnBu'), aspect=1, extent=extent, alpha=.3)
    time_slice = 0
    thrshld = 0.01
    wf_data = seis.wavefield
    wf_data_normalize = wf_data / np.amax(wf_data)
    n_frames = 50
    framerate = np.int(np.ceil(wf_data.shape[0] / n_frames))
    waves = wf_data_normalize[0::framerate, :, :]
    waves = waves[5, :, :]
    waves = np.ma.masked_where(np.abs(waves) <= thrshld, waves)

    plt.imshow(waves, **data_param)
    plt.imshow(seis.vp.T, **model_param)
    plt.show()

    plt.imshow(seis.wavefield[5,:,:].T, **data_param)
    plt.imshow(seis.vp.T, **model_param)
    plt.show()

    plt.imshow(seis.wavefield[200, :, :].T, **data_param)
    plt.imshow(seis.vp.T, **model_param)
    plt.show()

    plt.imshow(seis.wavefield[400, :, :].T, **data_param)
    plt.imshow(seis.vp.T, **model_param)
    plt.show()

