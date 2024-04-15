#! /usr/bin/env python3

# Take a point cloud as an input and 
#   return a panoramic depth image 

import numpy as np
import matplotlib.pyplot as plt


## Scale an array to a new range.
def scaling(array, depth_range=(0.3, 100), scale=255, dtype=np.uint8):
    """ Scale an array to a new range based on the min 
    and max values, as well as the new scale. """
    return (((array - depth_range[0]) / float(depth_range[1] - depth_range[0]))*scale).astype(dtype)


## Convert a point cloud to a depth image.
def pointcloud_to_depthimage(pointcloud,
                            V_RES=0.7,
                            H_RES=0.5,
                            V_FOV=(-15.0, 15.0),
                            depth_range=(0.3, 100)):
    """ Take a point cloud and convert it to a depth image with
        a 360 deg panoramic view, returned as a numpy array.
    Args:
        pointcloud: (np.array)
            The pointcloud should be a numpy array.
            The shape must be at lest Nx3. (allow intensity values)
            - Where N is the number of points.
            - Each point is specified by at least 3 values (x, y, z).
        V_RES: (float)
            The vertical angular resolution of the depth image in degrees.
        H_RES: (float)
            The horizontal angular resolution of the depth image in degrees.
        V_FOV: (tuple)
            The vertical field of view of the depth image.
        depth_range: (tuple)
            The depth range of the depth image.
    Returns:
        depth_image: (np.array)
            A panoramic 360 deg depth image as a numpy array. """

    ## 2D Projection
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
    d = np.sqrt(x**2 + y**2)

    ## Resolution settings
    V_FOV_total = -V_FOV[0] + V_FOV[1]

    ## Mapping cylinder
    x_img, y_img = np.arctan2(y, x) / (np.pi * H_RES / 180), \
                   np.arctan2(z, d) / (np.pi * V_RES / 180)

    ## Max width and height for image
    D_plane = (V_FOV_total / V_RES) / (V_FOV_total * (np.pi/180)) 
    H_below = D_plane * np.tan(-V_FOV[0] * (np.pi / 180))
    H_above = D_plane * np.tan(V_FOV[1] * (np.pi / 180))
    x_max, y_max = int(np.ceil(360.0 / H_RES)), \
                   int(np.ceil(H_above + H_below))

    ## Shift coordinates to make (0, 0) the minimum
    x_min, y_min = -360.0 / H_RES / 2, \
                   -(V_FOV[1] / V_RES)
    x_img, y_img = np.trunc(- x_img - x_min).astype(np.int32), \
                   np.trunc(- y_img - y_min).astype(np.int32)

    ## Clip distances
    d = np.clip(d, a_min=depth_range[0], a_max=depth_range[1])

    ## Create depth image
    image = np.zeros((y_max, x_max), dtype=np.uint8)
    image[y_img, x_img] = scaling(d, depth_range=depth_range)

    return image