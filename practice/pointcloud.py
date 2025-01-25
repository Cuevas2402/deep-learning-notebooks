# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
# %%
hw"""

import open3d as o3d
import numpy as np
import laspy
import pandas as pd
import matplotlib.pyplot as plt

# %%

las = laspy.read("./db/pcl/section2.las")
print(len(las.points))
print(list(las.point_format.dimension_names))

pcl = o3d.geometry.PointCloud()

points = np.vstack((las.x, las.y, las.z))
colors = np.vstack((las.red, las.green, las.blue)).astype(np.float64)
colors /= 65545

pcl.points = o3d.utility.Vector3dVector(points.transpose())
pcl.colors = o3d.utility.Vector3dVector(colors.transpose())
center = pcl.get_center()
pcl.translate(-center)
o3d.visualization.draw_geometries([pcl])


# %%

pcl = pcl.voxel_down_sample(voxel_size=0.5)
o3d.visualization.draw_geometries([pcl])
# %%

clss = np.unique(las.classification)
print(clss)


# %%

mask = las.classification == 2

xyz_t = np.vstack((las.x[mask], las.y[mask], las.z[mask]))

ground_pts = o3d.geometry.PointCloud()
ground_pts.points = o3d.utility.Vector3dVector(xyz_t.transpose())

o3d.visualization.draw_geometries([ground_pts])

# %%

notground = las.classification == 1
points = np.vstack((las.x[notground], las.y[notground], las.z[notground]))
#colors = np.vstack((las.red[notground], las.green[notground], las.blue[notground]))
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(points.transpose())
#pclnotground.colors = o3d.utility.Vector3dVector(colors.transpose())
o3d.visualization.draw_geometries([pcl])


# %%

nn_distance = np.mean(pcl.compute_nearest_neighbor_distance())
print(nn_distance)

# %%
#mask = las.classification == 1
#pcl = o3d.geometry.PointCloud()

#points = np.vstack((las.x[mask], las.y[mask], las.z[mask]))
#pcl.points = o3d.utility.Vector3dVector(points.transpose())

#pcl = pcl.voxel_down_sample(voxel_size=0.7)
epsilon = 2
min_cluster_points = 100
labels = np.array(pcl.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 1
pcl.colors = o3d.utility.Vector3dVector(colors[:,:3])
pcl
o3d.visualization.draw_geometries([pcl])


