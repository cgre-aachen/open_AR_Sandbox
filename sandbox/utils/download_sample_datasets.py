# %%
import os, sys
sys.path.append('.')
import pooch
from sandbox import _test_data, set_logger
from pooch import HTTPDownloader
download = HTTPDownloader(progressbar=True)
logger = set_logger(__name__)

#
# %%
parent_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC?path=%2F"
tests_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/download?path=%2FTests&files="
tests = ["arucos.pkl",
         "frame1.npz",
         "frame2.npz",
         "frame3.npz",
         "frame4.npz"]

topomodule_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/download?path=%2FTopoModule%2Fsaved_DEMs&files="
topomodule_files = ["1.npz",
                    "2.npz",
                    "3.npz",
                    "4.npz",
                    "DEM1.npz",
                    "DEM10.npz",
                    "DEM11.npz",
                    "DEM2.npz",
                    "DEM3.npz",
                    "DEM4.npz",
                    "DEM5.npz",
                    "DEM6.npz",
                    "DEM7.npz",
                    "DEM8.npz",
                    "DEM9.npz",
                    "Landslide_test_1.npz",
                    "bennisson_raster_DEM_04.npy",
                    "savedTopography.npz",
                    "test.npz"]

landslides_dems_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/" \
                      "download?path=%2FLanslideSimulation%2Fsaved_DEMs&files="
landslides_dems = ["Topography_0.npz",
                   "Topography_1.npz",
                   "Topography_2.npz",
                   "Topography_3.npz",
                   "Topography_4.npz"]
landslides_areas_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/" \
                       "download?path=%2FLanslideSimulation%2Fsaved_ReleaseAreas&files="
landslides_areas = ["ReleaseArea_0_1.npy",
                    "ReleaseArea_1_1.npy",
                    "ReleaseArea_1_2.npy",
                    "ReleaseArea_1_3.npy",
                    "ReleaseArea_2_1.npy",
                    "ReleaseArea_2_2.npy",
                    "ReleaseArea_2_3.npy",
                    "ReleaseArea_3_1.npy",
                    "ReleaseArea_3_2.npy",
                    "ReleaseArea_3_3.npy",
                    "ReleaseArea_4_1.npy",
                    "ReleaseArea_4_2.npy",
                    "ReleaseArea_4_3.npy",
                    ]
landslides_results_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/" \
                         "download?path=%2FLanslideSimulation%2Fsimulation_data&files="
landslides_results = ["Sim_Topo0_Rel01_results4sandbox.npz",
                      "Sim_Topo1_Rel11_results4sandbox.npz",
                      "Sim_Topo1_Rel12_results4sandbox.npz",
                      "Sim_Topo1_Rel13_results4sandbox.npz",
                      "Sim_Topo2_Rel21_results4sandbox.npz",
                      "Sim_Topo2_Rel22_results4sandbox.npz",
                      "Sim_Topo2_Rel23_results4sandbox.npz",
                      "Sim_Topo3_Rel31_results4sandbox.npz",
                      "Sim_Topo3_Rel32_results4sandbox.npz",
                      "Sim_Topo3_Rel33_results4sandbox.npz",
                      "Sim_Topo4_Rel41_results4sandbox.npz",
                      "Sim_Topo4_Rel42_results4sandbox.npz",
                      "Sim_Topo4_Rel43_results4sandbox.npz",
                      ]

landscape_models = ["Aletsch_1k",
                    "Aletsch_5k",
                    "AletschWin_3k",
                    "AletschWin_10k",
                    "Allgaeu_5k",
                    "AllgaeuSum_10k",
                    "AlpsSum1_1k",
                    "AlpsSum1_5k",
                    "AlpsSum1_10k",
                    "AlpsSum_3k",
                    "BernSum_5k",
                    "BernSum_10k",
                    "BernWin_5k",
                    "BernWin_10k"]
landscape_urls = ["https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/download?path=%2FLandscapeGeneration"
                  + "%2F" + i + "&files=" for i in landscape_models]

landscape_data = ["latest_net_D.pth",
                  "latest_net_G.pth",
                  "loss_log.txt",
                  "test_opt.txt",
                  "train_opt.txt"]

# Benisson model
gempy_benisson_url = "https://rwth-aachen.sciebo.de/s/oKxBxb1oGW2ZsoC/download?path=%2FGempy%2FBenisson_model&files="
gempy_benisson = ["Benisson_04_elev_contours.dbf",
                  "Benisson_04_elev_contours.prj",
                  "Benisson_04_elev_contours.shp",
                  "Benisson_04_elev_contours.shx",
                  "Benisson_Map_04.png",
                  "extent.dbf",
                  "extent.prj",
                  "extent.shp",
                  "extent.shx",
                  "interfaces_point.dbf",
                  "interfaces_point.prj",
                  "interfaces_point.shp",
                  "interfaces_point.shx",
                  "orientation.dbf",
                  "orientation.prj",
                  "orientation.shp",
                  "orientation.shx"
                  ]

gempy_example_models_url = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/jan_models/"
gempy_example_models = ["model1_orientations.csv",
                        "model1_surface_points.csv",
                        "model2_orientations.csv",
                        "model2_surface_points.csv",
                        "model3_orientations.csv",
                        "model3_surface_points.csv",
                        "model4_orientations.csv",
                        "model4_surface_points.csv",
                        "model5_orientations.csv",
                        "model5_surface_points.csv",
                        "model6_orientations.csv",
                        "model6_surface_points.csv",
                        "model7_orientations.csv",
                        "model7_surface_points.csv"
                        ]
gempy_example_models_url2 = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/lisa_models/"
gempy_example_models2 = ["foliations7.csv",
                         "interfaces7.csv",
                         ]


def create_pooch(base_url, files, target):
    """
    Create POOCH class to fetch files from a website
    Args:
        base_url: Base URL for the remote data source.
        files: A record of the files that are managed by this Pooch.
        target: The path to the local data storage folder
    Returns:
        POOCH class
    """
    pc = pooch.create(base_url=base_url,
                      path=target,
                      registry={i: None for i in files})  # None because the Hash is always changing.. Sciebo problem?
    logger.info("Pooch created for url: %s" % base_url)
    return pc


def download_test_data():
    """
    Download all data for testing
    Returns:

    """
    try:
        pooch_data = create_pooch(tests_url, tests, _test_data.get("test"))
        for file in tests:
            pooch_data.fetch(file, downloader=download)
        logger.info("Data for testing downloaded")
    except Exception as e:
        logger.error(e, exc_info=True)


def download_topography_data():
    """
    Download all available DEMs to SaveTopoModule
    Returns:

    """
    try:
        pooch_topo = create_pooch(topomodule_url, topomodule_files, _test_data.get("topo"))
        for file in topomodule_files:
            pooch_topo.fetch(file, downloader=download)
        logger.info("Data for topography downloaded")
    except Exception as e:
        logger.error(e, exc_info=True)

def download_landslides_data():
    """Download all available to display the landslide simulations """
    try:
        pooch_landslides_dem = create_pooch(landslides_dems_url, landslides_dems, _test_data.get("landslide_topo"))
        pooch_landslides_area = create_pooch(landslides_areas_url, landslides_areas, _test_data.get("landslide_release"))
        pooch_landslides_sim = create_pooch(landslides_results_url, landslides_results,
                                            _test_data.get("landslide_simulation"))

        for file in landslides_dems:
            pooch_landslides_dem.fetch(file, downloader=download)
        for file in landslides_areas:
            pooch_landslides_area.fetch(file, downloader=download)
        for file in landslides_results:
            pooch_landslides_sim.fetch(file, downloader=download)
        logger.info("Data for landslides downloaded")
    except Exception as e:
        logger.error(e, exc_info=True)


def download_benisson_model():
    """Dowload data for construction of Benisson model with gempy"""
    try:
        pooch_gempy = create_pooch(gempy_benisson_url, gempy_benisson, _test_data.get("gempy_data"))
        for file in gempy_benisson:
            pooch_gempy.fetch(file, downloader=download)
        logger.info("Data for benisson model downloaded")
    except Exception as e:
        logger.error(e, exc_info=True)

def download_example_gempy_model():
    """Dowload data for construction of example models with gempy"""
    try:
        pooch_gempy_example = create_pooch(gempy_example_models_url, gempy_example_models, _test_data.get("gempy_example_data"))
        pooch_gempy_example2 = create_pooch(gempy_example_models_url2, gempy_example_models2, _test_data.get("gempy_example_data"))
        for file in gempy_example_models:
            pooch_gempy_example.fetch(file, downloader=download)
        for file in gempy_example_models2:
            pooch_gempy_example2.fetch(file, downloader=download)
        logger.info("Data for gempy example models downloaded")
    except Exception as e:
        logger.error(e, exc_info=True)

def download_landscape_name(name_model: str):
    """Download an specific trained model"""
    try:
        if name_model not in landscape_models:
            logger.warning("\n Model with name '%s' not available for download. "
                            "\n Available models are %s" % (name_model, str(landscape_models)))
            return False

        loc_dir = _test_data.get("landscape_generation") + "checkpoints" + os.sep + name_model
        if not os.path.isdir(loc_dir):
            os.mkdir(loc_dir)

        pos = [i for i in range(len(landscape_models)) if name_model == landscape_models[i]][0]
        pooch_landscape = create_pooch(landscape_urls[pos], landscape_data, loc_dir)

        for file in landscape_data:
            pooch_landscape.fetch(file, downloader=download)
        logger.info("Model %s downloaded for landscape generation" % name_model)

    except Exception as e:
        logger.error(e, exc_info=True)

def download_landscape_all():
    """Download all trained models available"""
    for name_model in landscape_models:
        download_landscape_name(name_model)


#%%
if __name__ == '__main__':
    if input("Do you want to download the genpy data for the example models? (12 kB) [y/n]") == "y":
        download_example_gempy_model()

    if input("Do you want to download the Test data? (25.4 MB) [y/n]") == "y":
        download_test_data()

    if input("Do you want to download some DEMs to the SaveLoadModule? (24 MB) [y/n]") == "y":
        download_topography_data()

    if input("Do you want to download the Landslide data to the LandslideModule? (114 MB) [y/n]") == "y":
        download_landslides_data()

    if input("Do you want to download the Gempy data for the Benisson Model? (1.2 MB) [y/n]") == "y":
        download_benisson_model()

    if input("Do you want to download all the Trained models for the LandscapeGeneration module? (3 GB) [y/n]") == \
            "y":
        if input("Are you sure? All of them? It is 3 GB of data [y/n]") == "y":
            download_landscape_all()
    while True:
        if input("Do you want to download an specific Trained model? [y/n]") == "y":
            print("Available models: %s" % landscape_models)
            model = input("Name of model to download:")
            download_landscape_name(model)
        else:
            break
