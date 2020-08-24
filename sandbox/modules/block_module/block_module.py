import numpy
import pickle
import matplotlib
import skimage
import panel as pn
from sandbox.modules.template import ModuleTemplate

class BlockModule(ModuleTemplate):
    # child class of Model

    def __init__(self, calibrationdata, sensor, projector, crop=True, **kwarg):
        super().__init__(calibrationdata, sensor, projector, crop, **kwarg)  # call parent init
        self.block_dict = {}
        self.cmap_dict = {}
        self.displayed_dataset_key = "mask"  # variable to choose displayed dataset in runtime
        self.rescaled_block_dict = {}
        self.reservoir_topography = None
        self.rescaled_reservoir_topography = None
        self.show_reservoir_topo = False
        self.num_contours_reservoir_topo = 10  # number of contours in
        self.reservoir_topography_topo_levels = None  # set in setup and in widget.
        self.result = None  # stores the output array of the current frame

        # #rescaled Version of Livecell information. masking has to be done after scaling because the scaling does not support masked arrays
        # self.rescaled_data_mask = None
        self.index = None  # index to find the cells in the rescaled block modules, corresponding to the topography in the sandbox
        self.widget = None  # widget to change models in runtime
        self.min_sensor_offset = 0
        self.max_sensor_offset = 0
        self.minmax_sensor_offset = 0
        self.original_sensor_min = 0
        self.original_sensor_max = 0
        self.mask_threshold = 0.5  # set the threshold for the mask array, interpolated between 0.0 and 1.0 #obsolete!

        self.num_contour_steps = 20

    def setup(self):
        if self.block_dict is None:
            print("No model loaded. Load a model first with load_module_vip(infile)")
            pass
        elif self.cmap_dict is None:
            self.set_colormaps()
        self.rescale_blocks()
        # self.rescale_mask() #nearest neighbour? obsolete! mask is now part of the block_dict

        self.displayed_dataset_key = list(self.block_dict)[1]

        self.plot.contours_color = 'w'  # Adjust default contour color

        self.projector.frame.object = self.plot.figure  # Link figure to projector

        self.calculate_reservoir_contours()

    def update(self):
        # with self.lock:
        frame = self.sensor.get_frame()

        if self.crop is True:
            frame = self.crop_frame(frame)
        depth_mask = self.depth_mask(frame)

        ###workaround:resize depth mask
        # depth_mask = skimage.transform.resize(
        #    depth_mask,
        #    (
        #    self.block_dict[self.displayed_dataset_key].shape[0], self.block_dict[self.displayed_dataset_key].shape[1]),
        #    order=0
        # )

        frame = self.clip_frame(frame)

        ##workaround: reshape frame to array size, not the other way around!
        #  frame = skimage.transform.resize(
        #          frame,
        #          (self.block_dict[self.displayed_dataset_key].shape[0], self.block_dict[self.displayed_dataset_key].shape[1]),
        #          order=1
        #  )

        if self.displayed_dataset_key is 'mask':  # check if there is a data_mask, TODO: try except key error
            data = self.rescaled_block_dict[self.displayed_dataset_key]
        #  data = self.block_dict[self.displayed_dataset_key]
        else:  # apply data mask

            data = numpy.ma.masked_where(self.rescaled_block_dict['mask'] < self.mask_threshold,
                                         self.rescaled_block_dict[self.displayed_dataset_key]
                                         )

        zmin = self.calib.s_min
        zmax = self.calib.s_max

        index = (frame - zmin) / (zmax - zmin) * (data.shape[2] - 1.0)  # convert the z dimension to index
        index = index.round()  # round to next integer
        self.index = index.astype('int')

        # querry the array:
        i, j = numpy.indices(data[..., 0].shape)  # create arrays with the indices in x and y
        self.result = data[i, j, self.index]

        self.result = numpy.ma.masked_array(self.result, mask=depth_mask)  # apply the depth mask

        self.plot.ax.cla()

        self.plot.vmin = zmin
        self.plot.vmax = zmax
        cmap = self.cmap_dict[self.displayed_dataset_key][0]
        cmap.set_over('black')
        cmap.set_under('black')
        cmap.set_bad('black')

        norm = self.cmap_dict[self.displayed_dataset_key][1]
        min = self.cmap_dict[self.displayed_dataset_key][2]
        max = self.cmap_dict[self.displayed_dataset_key][3]
        self.plot.cmap = cmap
        self.plot.norm = norm
        self.plot.render_frame(self.result, contourdata=frame, vmin=min, vmax=max)  # plot the current frame

        if self.show_reservoir_topo is True:
            self.plot.ax.contour(self.rescaled_reservoir_topography, levels=self.reservoir_topography_topo_levels)
        # render and display
        # self.plot.ax.axis([0, self.calib.s_frame_width, 0, self.calib.s_frame_height])
        # self.plot.ax.set_axis_off()

        self.projector.trigger()
        # return True

    def load_model(self, model_filename):
        """
        loads a regular grid dataset parsed and prepared with the RMS Grid class.
        the pickled list contains 2 entries:
        1.  The regridded Block dictionary
        2.  a 2d array of the lateral size of the blocks with the z values of the uppermost layer
            (= the shape of the reservoir top surface)
        Args:
            model_filename: string with the path to the file to load

        Returns: nothing, changes in place the

        """
        data_list = pickle.load(open(model_filename, "rb"))
        self.block_dict = data_list[0]
        self.reservoir_topography = data_list[1]
        print('Datasets loaded: ', self.block_dict.keys())

    def create_cmap(self, clist):
        """
        create a matplotlib colormap object from a list of discrete colors
        :param clist: list of colors
        :return: colormap
        """

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('default', clist, N=256)
        return cmap

    def create_norm(self, vmin, vmax):
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return norm

    def set_colormap(self, key=None, cmap='jet', norm=None):
        min = numpy.nanmin(self.block_dict[key].ravel())  # find min ignoring NaNs
        max = numpy.nanmax(self.block_dict[key].ravel())

        if isinstance(cmap, str):  # get colormap by name
            cmap = matplotlib.cm.get_cmap(name=cmap, lut=None)

        if norm is None:
            norm = self.create_norm(min, max)

        self.cmap_dict[key] = [cmap, norm, min, max]

    def set_colormaps(self, cmap=None, norm=None):
        """
        iterates over all datasets and checks if a colormap has been set. if no colormaps exists it creates one.
        default colormap: jet
        :param cmap:
        :param norm:
        :return:
        """
        for key in self.block_dict.keys():
            if key not in self.cmap_dict.keys():  # add entry if not already in cmap_dict
                self.set_colormap(key)

    def rescale_blocks(self):  # scale the blocks xy Size to the cropped size of the sensor
        for key in self.block_dict.keys():
            rescaled_block = skimage.transform.resize(
                self.block_dict[key],
                (self.calib.s_frame_height, self.calib.s_frame_width),
                order=0
            )

            self.rescaled_block_dict[key] = rescaled_block

        if self.reservoir_topography is not None:  # rescale the topography map
            self.rescaled_reservoir_topography = skimage.transform.resize(
                self.reservoir_topography,
                (self.calib.s_frame_height, self.calib.s_frame_width),
                order=0  # nearest neighbour
            )

    def rescale_mask(self):  # scale the blocks xy Size to the cropped size of the sensor
        rescaled_mask = skimage.transform.resize(
            self.data_mask,
            (self.calib.s_frame_height, self.calib.s_frame_width),
            order=0
        )
        self.rescaled_data_mask = rescaled_mask

    def clear_models(self):
        self.block_dict = {}

    def clear_rescaled_models(self):
        self.rescaled_block_dict = {}

    def clear_cmaps(self):
        self.cmap_dict = {}

    def calculate_reservoir_contours(self):
        min = numpy.nanmin(self.rescaled_reservoir_topography.ravel())
        max = numpy.nanmax(self.rescaled_reservoir_topography.ravel())
        step = (max - min) / float(self.num_contours_reservoir_topo)
        print(min, max, step)
        self.reservoir_topography_topo_levels = numpy.arange(min, max, step=step)

    def widget_mask_threshold(self):
        """
        displays a widget to adjust the mask threshold value

        """
        pn.extension()
        widget = pn.widgets.FloatSlider(name='mask threshold (values smaller than the set threshold will be masked)',
                                        start=0.0, end=1.0, step=0.01, value=self.mask_threshold)

        widget.param.watch(self._callback_mask_threshold, 'value', onlychanged=False)

        return widget

    def _callback_mask_threshold(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.mask_threshold = event.new
        self.resume()

    def show_widgets(self):
        self.original_sensor_min = self.calib.s_min  # store original sensor values on start
        self.original_sensor_max = self.calib.s_max

        widgets = pn.WidgetBox(self._widget_model_selector(),
                               self._widget_sensor_top_slider(),
                               self._widget_sensor_bottom_slider(),
                               self._widget_sensor_position_slider(),
                               self._widget_show_reservoir_topography(),
                               self._widget_reservoir_contours_num(),
                               self._widget_contours_num()
                               )

        panel = pn.Column("### Interaction widgets", widgets)
        self.widget = panel
        return panel

    def _widget_model_selector(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.RadioButtonGroup(name='Model selector',
                                             options=list(self.block_dict.keys()),
                                             value=self.displayed_dataset_key,
                                             button_type='success')

        widget.param.watch(self._callback_selection, 'value', onlychanged=False)

        return widget

    def _callback_selection(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.displayed_dataset_key = event.new
        self.resume()

    def _widget_sensor_top_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset top of the model ', start=-250, end=250, step=1, value=0)

        widget.param.watch(self._callback_top_slider, 'value', onlychanged=False)

        return widget

    def _callback_top_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.min_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_sensor_bottom_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset bottom of the model ', start=-250, end=250, step=1, value=0)

        widget.param.watch(self._callback_bottom_slider, 'value', onlychanged=False)

        return widget

    def _callback_bottom_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.max_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_sensor_position_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset the model in vertical direction ', start=-250, end=250, step=1,
                                      value=0)

        widget.param.watch(self._callback_position_slider, 'value', onlychanged=False)

        return widget

    def _callback_position_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.minmax_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _update_sensor_calib(self):
        self.calib.s_min = self.original_sensor_min + self.min_sensor_offset + self.minmax_sensor_offset
        self.calib.s_max = self.original_sensor_max + self.max_sensor_offset + self.minmax_sensor_offset

    def _widget_show_reservoir_topography(self):
        widget = pn.widgets.Toggle(name='show reservoir top contours',
                                   value=self.show_reservoir_topo,
                                   button_type='success')
        widget.param.watch(self._callback_show_reservoir_topography, 'value', onlychanged=False)

        return widget

    def _callback_show_reservoir_topography(self, event):
        self.pause()
        self.show_reservoir_topo = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_reservoir_contours_num(self):
        """ Shows a widget that allows to change the contours step size"""

        widget = pn.widgets.IntSlider(name='number of contours in the reservoir topography',
                                      start=0,
                                      end=100,
                                      step=1,
                                      value=round(self.num_contours_reservoir_topo))

        widget.param.watch(self._callback_reservoir_contours_num, 'value', onlychanged=False)
        return widget

    def _callback_reservoir_contours_num(self, event):
        self.pause()
        self.num_contours_reservoir_topo = event.new
        self.calculate_reservoir_contours()
        self.resume()

    def _widget_contours_num(self):
        """ Shows a widget that allows to change the contours step size"""

        widget = pn.widgets.IntSlider(name='number of contours in the sandbox',
                                      start=0,
                                      end=100,
                                      step=1,
                                      value=self.num_contour_steps)

        widget.param.watch(self._callback_contours_num, 'value', onlychanged=False)
        return widget

    def _callback_contours_num(self, event):
        self.pause()
        self.plot.vmin = self.calib.s_min
        self.plot.vmax = self.calib.s_max
        self.num_contour_steps = event.new
        self.plot.contours_step = (self.plot.vmax - self.plot.vmin) / float(self.num_contour_steps)
        self.resume()
