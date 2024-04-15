import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


## Visualize
def visualize(point_cloud, object=[], plot_second_on_top=False,
                point_size=20, point_size_2=100, color='C0', color2='r'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], color=color, s=point_size)
    if plot_second_on_top:
        ax.scatter(object[:,0], object[:,1], object[:,2], color=color2, s=point_size_2)
    ax.set_axis_off()
    plt.show()

