from sandbox import _test_data
from sandbox.modules import TopoModule
import matplotlib.pyplot as plt
import pytest
import numpy as np

def load_marker():
    import pandas as pd
    df = pd.DataFrame({"box_x": [25], "box_y": [120], "is_inside_box": [True]})
    #arucos = _test_data['test'] + "arucos.pkl"
    #try:
    #    df = pd.read_pickle(arucos)
    ##    print("Arucos loaded")
    #except:
    #    df = pd.DataFrame()
    #    print("No arucos found")

    return df

file = np.load(_test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
frame = frame + np.abs(frame.min())
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': load_marker(),
                    'cmap': plt.cm.get_cmap('gist_earth_r'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}


def test_init():
    module = TopoModule()
    print(module)


def test_update():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_update_no_sea():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.sea = False
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_update_no_normalized():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.normalize = False
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_widgets():
    module = TopoModule(extent=extent)
    widgets = module.show_widgets()
    widgets.show()


def test_add_countour():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()
    #pytest.sb_params['frame'] = frame
    module.sea = True
    module.sea_contour = True
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()
    pytest.sb_params['ax'].cla()
    module.sea_contour = False
    #pytest.sb_params['frame'] = frame
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()

def test_normalize_negative_height():
    module = TopoModule(extent=extent)
    plt.imshow(pytest.sb_params['frame'], origin="lower", cmap = "viridis")
    plt.colorbar()
    plt.show()
    n = -200
    new_frame_negative, new_extent_negative = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height= n)
    assert new_frame_negative.min() == n
    plt.imshow(new_frame_negative, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()
    p=200
    new_frame_positive, new_extent_positive = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height=p)
    assert new_frame_positive.min() == p
    plt.imshow(new_frame_positive, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()

    c = 0
    new_frame_positive, new_extent_positive = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height=c)
    assert new_frame_positive.min() == c
    plt.imshow(new_frame_positive, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()

def test_single_path():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.sea = True
    module.side_flooding = True
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']

    ax.imshow(sb_params.get('frame'),  # vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    ax.plot(module._marker_contour_val[0], module._marker_contour_val[1], 'r*')
    fig.show()


def test_solve_maze():
    # set We create a matrix with zeros of the same size
    # Put a 1 to the starting point
    # Everywhere around 1 we put 2 , if there is no wall
    # Everywhere around 2 we put 3 , if there is no wall
    # and so on…
    # once we put a number at the ending point, we stop. This number is actually the minimal path length
    a = frame#[:50, :50]
    m = np.zeros((a.shape))
    m[140,150] = 1
    # FUnction for just one step
    tolerance = 0#.5
    def make_step(k):
        fin = False
        for i in range(len(m)):
            for j in range(len(m[i])):
                if m[i][j] == k:
                    z = frame[i][j] + tolerance  # Current height check if is below
                    if i > 0 and m[i - 1][j] == 0 and frame[i - 1][j] <= z:  # down
                        m[i - 1][j] = k + 1
                        fin = True
                    if j > 0 and m[i][j - 1] == 0 and frame[i][j - 1] <= z:  # left
                        m[i][j - 1] = k + 1
                        fin = True
                    if i < len(m) - 1 and m[i + 1][j] == 0 and frame[i + 1][j] <= z:  # up
                        m[i + 1][j] = k + 1
                        fin = True
                    if j < len(m[i]) - 1 and m[i][j + 1] == 0 and frame[i][j + 1] <= z:  # right
                        m[i][j + 1] = k + 1
                        fin = True
        return fin
    k = 0
    work = True
    while work:
        k += 1
        work = make_step(k)
    #[make_step(i+1) for i in range(10)]
    m[m==0] = np.nan
    plt.imshow(a, cmap = "gist_earth")
    plt.imshow(m)
    plt.show()

def test_solve_maze_with_gradients():
    a = frame[:50, :50]
    m = np.zeros((a.shape))
    m[140, 150] = 1
    # FUnction for just one step
    tolerance = 0.5

    dx, dy = np.gradient(frame)
