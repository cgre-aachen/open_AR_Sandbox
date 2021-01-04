import os
import gempy as gp

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
all_models = ['Horizontal_layers',
              'Recumbent_fold',
              'Anticline',
              'Pinchout',
              'Fault',
              'Unconformity']
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"


def create_model_dict(model_name: list = all_models, **kwargs):
    """
    Create all the example models for the gempy module and stores them in a dictionary
    Returns:
        Dictionary with all the gempy models
    """
    model_dict = {}
    for model in model_name:
        model_dict.update({model: create_example_model(model, **kwargs)})
    return model_dict


def create_example_model(name, extent=[0, 1000, 0, 1000, 0, 1800], do_sections=False,
                         change_color=False, data_path=path_to_data, resolution=[20, 20, 20],
                         theano_optimizer='fast_compile'):
    # _test_data['gempy_data'], theano_optimizer='fast_compile'):
    """
    Create all the example models
    Args:
        name: Name of the model to create
        extent: extent of the model in model units
        do_sections: False
        change_color: False
        data_path: Path to the .csv files storing the information
        theano_optimizer: 'fast_compile'
    Returns:
        geo_model

    """
    assert name in all_models, 'possible model names are ' + str(all_models)
    if name == 'Horizontal_layers':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model1_orientations.csv",
                                   path_i=data_path + "model1_surface_points.csv")

        gp.map_stack_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#9f0052', 'rock1': '#e36746',
                                                     'basement': '#f9f871'})

    elif name == 'Anticline':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model2_orientations.csv",
                                   path_i=data_path + "model2_surface_points.csv")
        gp.map_stack_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                             "Basement_Series": ('basement')})

    elif name == 'Recumbent_fold':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model3_orientations.csv",
                                   path_i=data_path + "model3_surface_points.csv")
        gp.map_stack_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                             "Basement_Series": ('basement')})
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#e36746', 'rock1': '#c0539f',
                                                     'basement': '#006fa8'})

    elif name == 'Pinchout':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model4_orientations.csv",
                                   path_i=data_path + "model4_surface_points.csv")
        gp.map_stack_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                             "Basement_Series": ('basement')})
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#a1b455', 'rock1': '#ffbe00',
                                                     'basement': '#006471'})

    elif name == 'Fault':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model5_orientations.csv",
                                   path_i=data_path + "model5_surface_points.csv")
        gp.map_stack_to_surfaces(geo_model, {"Fault_Series": 'fault',
                                             "Strat_Series": ('rock2', 'rock1')})
        geo_model.set_is_fault(['Fault_Series'], change_color=False)
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#00c2d0', 'rock1': '#a43d00',
                                                     'basement': '#76a237', 'fault': '#000000'})
    elif name == 'Unconformity':
        geo_model = gp.create_data(name, extent=extent, resolution=resolution,
                                   path_o=data_path + "model6_orientations.csv",
                                   path_i=data_path + "model6_surface_points.csv")

        gp.map_stack_to_surfaces(geo_model, {"Strat_Series1": ('rock3'),
                                             "Strat_Series2": ('rock2', 'rock1'),
                                             "Basement_Series": ('basement')})
    else:
        raise NotImplementedError(name, "Not available in example models")

    if do_sections:
        geo_model.set_section_grid({'section' + ' ' + name: ([0, 500], [1000, 500], [30, 30])})

    interp_data = gp.set_interpolator(geo_model,  # compile_theano=True,
                                      theano_optimizer=theano_optimizer)

    sol = gp.compute_model(geo_model, compute_mesh=False)

    if do_sections:
        gp.plot_2d(geo_model, section_names=['section' + ' ' + name], show_data=False)

    return geo_model
