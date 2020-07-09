import numpy
import matplotlib
import matplotlib.pyplot as plt


class CmapModule:
    """
    Class to manage changes in the colormap and plot in the desired projector figure
    """
    def __init__(self, cmap='gist_earth_r', over=None, under=None,
                 bad=None, norm=None, lot=None, vmin=None, vmax=None,
                 extent=None):
        """
        Initialize the colormap to plot using pcolormesh()
        Args:
            cmap (str or plt.Colormap): Matplotlib colormap, given as name or instance.
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.
            norm: Future feature!
            lot: Future feature!
            vmin (float): ...
            vmax (float): ...
            extent (list): ...
        """
        # z-range handling
        self.extent = extent
        if vmin is not None:
            self.vmin = vmin
        else:
            self.vmin = self.extent[4]

        if vmax is not None:
            self.vmax = vmax
        else:
            self.vmax = self.extent[5]

        self.cmap = self.set_cmap(plt.cm.get_cmap(cmap), over, under, bad)

        self.norm = norm  # TODO: Future feature
        self.lot = lot  # TODO: Future feature
        self.col = None

    def set_cmap(self, cmap, over=None, under=None, bad=None):
        """
        Methods to mask the values outside the extent
        Args:
            cmap: (matplotlib colormap): colormap to use
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.

        Returns:
        """
        if over is not None:
            cmap.set_over(over, 1.0)
        if under is not None:
            cmap.set_under(under, 1.0)
        if bad is not None:
            cmap.set_bad(bad, 1.0)
        self.cmap = cmap
        return self.cmap

    def render_frame(self, data, ax, vmin=None, vmax=None):
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        self.col = ax.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=self.cmap, norm=self.norm)


