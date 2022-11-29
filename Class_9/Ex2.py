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

    pcd = o3d.io.read_point_cloud("./data/factory.ply")

    plane_model, inliers_idx = pcd.segment_plane(distance_threshold=0.3,
                                            ransac_n=3,
                                            num_iterations=100)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers_idx)
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) #paints the plain in red
    outlier_cloud = pcd.select_by_index(inliers_idx, invert=True)



    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('./data/factory_without_gound.ply',outlier_cloud,write_ascii=False,compressed=False,print_progress=False)


if __name__ == '__main__':
    main()