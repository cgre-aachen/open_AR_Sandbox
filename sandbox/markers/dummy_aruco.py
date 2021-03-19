import pandas as pd


def dummy_markers_in_frame(dict_position: dict, depth):
    """ Function to create a point to point map of the spatial/pixel equivalences between the depth space,
     color space and camera space.
     This method requires the depth frame to assign a depth value to the color point.
    Returns:
        CoordinateMap: DataFrame with the x,y,z values of the depth frame; x,y equivalence
        between the depth space to camera space and real world values of x,y and z in meters
    """
    height, width = depth.shape
    labels = {"ids",
              "depth_x",
              "depth_y",
              "box_x",
              "box_y",
              "is_inside_box",
              "counter"}
    df = pd.DataFrame(columns=labels)
    for ids in dict_position.keys():
        x, y = dict_position[ids]
        df_temp = pd.DataFrame(
            {'ids': [ids],
             'depth_x': [x],
             'depth_y': [y],
             'depth_z': [depth[y][x] if 0 < x < width and 0 < y < height else 0],
             'box_x': [x],
             'box_y': [y],
             'is_inside_box': [True if 0 < x < width and 0 < y < height else False],
             'counter': [0]
             }
        )
        df = pd.concat([df, df_temp], sort=False)
    df = df.set_index(df['ids'], drop=True)
    df = df.drop(columns=['ids'])
    return df
