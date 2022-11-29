#!/usr/bin/env python3

from pydoc import locate
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.48569235818388873, -0.81272016672538483, 0.32185223907817695 ],
			"lookat" : [ -5.2515106012699571, -1.2331649608099793, 4.1283505620774772 ],
			"up" : [ -0.20703638500269855, 0.25076680911690946, 0.9456489532222504 ],
			"zoom" : 0.3412
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def main():
    point_cloud_original = o3d.io.read_point_cloud("./data/factory_without_ground.ply")
    #point_cloud=deepcopy(point_cloud_original)
    point_cloud_down_sample= point_cloud_original.voxel_down_sample(voxel_size=0.3)

    cluster_idxs = np.array(point_cloud_down_sample.cluster_dbscan(eps=0.5, min_points=50, print_progress=True))

    possible_values= list(set(cluster_idxs))
    possible_values.remove(-1)
    largest_cluster_num_points=0
    largest_cluster_idx=0
    for value in possible_values:
        num_points=cluster_idxs.count(value)
        if num_points>largest_cluster_num_points:
            largest_cluster_idx=value
            largest_cluster_num_points=num_points

    largest_idxs=list(locate(cluster_idxs,lambda x: x==largest_cluster_idx))
    cloud_building=point_cloud_down_sample.select_by_index(largest_idxs)
    cloud_others=point_cloud_down_sample.select_by_index(largest_idxs,invert=True)

    cloud_others.paint_uniform_color(0,0,1)
    
    entities=[cloud_building,cloud_others]
    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])


if __name__ == '__main__':
    main()