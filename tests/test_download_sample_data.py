from sandbox.utils.download_sample_datasets import *
cache = pooch.os_cache("sandbox")

def test_download_test_data():
    pooch_data = create_pooch(tests_url, tests, cache)
    for file in tests:
        pooch_data.fetch(file, downloader=download)

def test_download_savetopo():
    pooch_topo = create_pooch(topomodule_url, topomodule_files, cache)
    for file in topomodule_files:
        pooch_topo.fetch(file, downloader=download)

def test_download_landslides():
    pooch_landslides_dem = create_pooch(landslides_dems_url, landslides_dems, cache)
    pooch_landslides_area = create_pooch(landslides_areas_url, landslides_areas, cache)
    pooch_landslides_sim = create_pooch(landslides_results_url, landslides_results, cache)

    for file in landslides_dems:
        pooch_landslides_dem.fetch(file, downloader=download)
    for file in landslides_areas:
        pooch_landslides_area.fetch(file, downloader=download)
    for file in landslides_results:
        pooch_landslides_sim.fetch(file, downloader=download)

def test_download_gempy():
    pooch_gempy = create_pooch(gempy_benisson_url, gempy_benisson, cache)
    for file in gempy_benisson:
        pooch_gempy.fetch(file, downloader=download)

def test_download_landscape():
    del landscape_data[1] # The most heavy file
    for name_model in landscape_models:

        pos = [i for i in range(len(landscape_models)) if name_model == landscape_models[i]][0]
        pooch_landscape = create_pooch(landscape_urls[pos], landscape_data, pooch.os_cache("sandbox/"+name_model))

        for file in landscape_data:
            pooch_landscape.fetch(file, downloader=download)