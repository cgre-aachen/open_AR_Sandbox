import numpy
import matplotlib.colors as mcolors

def plot_gempy(ax, geo_model):
    """
    Plot the geological map of the sandbox in the axes
    Args:
        ax: axes to the figure to plot
        geo_model: gempy model
    Returns:
        Painted axes

    """
    cmap = mcolors.ListedColormap(list(geo_model.surfaces.df['color']))
    ax.cla()
    ax = add_faults(ax, geo_model, cmap)
    ax = add_lith(ax, geo_model, cmap)
    return ax, cmap

def add_faults(ax, geo_model, cmap):
    ax = extract_boundaries(ax, geo_model, cmap, e_faults=True, e_lith=False)
    return ax

def add_lith(ax, geo_model, cmap):
    ax = extract_boundaries(ax, geo_model, cmap, e_faults=False, e_lith=True)
    return ax

def extract_boundaries(ax, geo_model, cmap, e_faults=False, e_lith=False):
    faults = list(geo_model._faults.df[geo_model._faults.df['isFault'] == True].index)
    shape = geo_model._grid.topography.resolution
    a = geo_model.solutions.geological_map[1]
    extent = geo_model._grid.topography.extent
    zorder = 2
    counter = a.shape[0]

    if e_faults:
        counters = numpy.arange(0, len(faults), 1)
        c_id = 0  # color id startpoint
    elif e_lith:
        counters = numpy.arange(len(faults), counter, 1)
        c_id = len(faults)  # color id startpoint
    else:
        raise AttributeError

    for f_id in counters:
        block = a[f_id]
        level = geo_model.solutions.scalar_field_at_surface_points[f_id][numpy.where(
            geo_model.solutions.scalar_field_at_surface_points[f_id] != 0)]

        levels = numpy.insert(level, 0, block.max())
        c_id2 = c_id + len(level)
        if f_id == counters.max():
            levels = numpy.insert(levels, level.shape[0], block.min())
            c_id2 = c_id + len(levels)  # color id endpoint
        block = block.reshape(shape)
        zorder = zorder - (f_id + len(level))

        if f_id >= len(faults):
            ax.contourf(block, 0, levels=numpy.sort(levels), colors=cmap.colors[c_id:c_id2][::-1],
                             linestyles='solid', origin='lower',
                             extent=extent, zorder=zorder)
        else:
            ax.contour(block, 0, levels=numpy.sort(levels), colors=cmap.colors[c_id:c_id2][0],
                            linestyles='solid', origin='lower',
                            extent=extent, zorder=zorder)
        c_id += len(level)

    return ax
