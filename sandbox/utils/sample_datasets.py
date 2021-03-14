# %%
import pooch
import os
from sandbox import _test_data
from warnings import warn
from pooch import HTTPDownloader

download = HTTPDownloader(progressbar=True)
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
                    "Aletsch_3k_lr0.002",
                    "AletschWin_10k",
                    "Allgaeu_5k",
                    "AllgaeuSum_10k",
                    "AlpsSum1_1k",
                    "AlpsSum1_5k",
                    "AlpsSum1_10k",
                    "AlpsSum_3k_lr0.002",
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


def create_pooch(base_url, files, target):
    pc = pooch.create(base_url=base_url,
                      path=target,
                      registry={i: None for i in files})  # None because the Hash is always changing.. Sciebo problem?
    return pc


def download_test_data():
    pooch_data = create_pooch(tests_url, tests, _test_data.get("test"))
    for file in tests:
        pooch_data.fetch(file, downloader=download)


def download_topography_data():
    pooch_topo = create_pooch(topomodule_url, topomodule_files, _test_data.get("topo"))
    for file in topomodule_files:
        pooch_topo.fetch(file, downloader=download)


def download_landslides_data():
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


def download_landscape_name(name_model: str):
    if name_model not in landscape_models:
        return warn("\n Model with name '%s' not available for download. "
                    "\n Available models are %s" % (name_model, str(landscape_models)))

    loc_dir = _test_data.get("landscape_generation") + "checkpoints" + os.sep + name_model
    if not os.path.isdir(loc_dir):
        os.mkdir(loc_dir)

    pos = [i for i in range(len(landscape_models)) if name_model == landscape_models[i]][0]
    pooch_landscape = create_pooch(landscape_urls[pos], landscape_data, loc_dir)

    for file in landscape_data:
        pooch_landscape.fetch(file, downloader=download)


def download_landscape_all():
    for name_model in landscape_models:
        download_landscape_name(name_model)


if __name__ == '__main__':
    download_test_data()
    download_topography_data()
    download_landslides_data()
    download_landscape_all()
