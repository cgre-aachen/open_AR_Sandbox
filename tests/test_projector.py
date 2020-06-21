#%%
import sandbox as sb
calib_dir = sb._calibration_dir

def test_init_projector():
    projector = sb.Projector(use_panel=False)
    assert projector.panel is not None

def test_save_load_calibration_projector():
    projector = sb.Projector(use_panel=False)
    file = calib_dir + 'test_projector_calibration.json'
    projector.save_json(file=file)
    # now to test if it loads correctly the saved one
    projector2 = sb.Projector(calibprojector = file, use_panel=False)

def test_open_panel_browser():
    projector = sb.Projector(use_panel=False)
    projector.start_server()

def test_delete_ax_image():
    projector = sb.Projector(use_panel=False)
    projector.ax.plot([10, 20, 30], [20, 39, 48])
    projector.clear_axes()

def test_delete_points():
    projector = sb.Projector(use_panel=False)
    line1 = projector.ax.plot([10, 20, 30], [20, 39, 48])
    projector.trigger()
    del line1
    projector.trigger()

