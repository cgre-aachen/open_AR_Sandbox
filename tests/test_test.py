import sandbox as sb
import matplotlib.pyplot as plt
import numpy as np

def test_data_path():
    print(sb._test_data)


def test_how_many_axis():
    fig, ax = plt.subplots()
    array = np.ones((10,10))
    array[:5] = 3
    ax.imshow(array)
    ax.imshow(array)
    ax.imshow(array)
    del ax.images[0]
    ax.imshow(array)

    fig.show()

