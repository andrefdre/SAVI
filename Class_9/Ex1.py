#!/usr/bin/env python3

import open3d as o3d
import numpy as np

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

    print("Load a ply point cloud, print it, and render it")
    #ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud("./data/factory.ply")
    print(point_cloud)
    print(np.asarray(point_cloud.points))
    o3d.visualization.draw_geometries([point_cloud],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])



if __name__ == '__main__':
    main()