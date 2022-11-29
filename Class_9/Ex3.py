#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import numpy as np
import matplotlib as mpl

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

class PlaneDetection():
    def __init__(self,point_cloud):
        self.point_cloud=point_cloud

    def colorize_inliers(self,r,g,b):
        self.inlier_cloud.paint_uniform_color([r, g, b]) #paints the plain in red


    def segment(self,distance_threshold=0.25,ransac_n=3,num_iterations=100):
        plane_model, inliers_idx = self.point_cloud.segment_plane(distance_threshold=distance_threshold,ransac_n=ransac_n,num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model
        self.inlier_cloud = self.point_cloud.select_by_index(inliers_idx)
        outlier_cloud = self.point_cloud.select_by_index(inliers_idx, invert=True)
        return outlier_cloud

    def __str__(self):
        return f"Plane equation: {self.a:.2f}x + {self.b:.2f}y + {self.c:.2f}z + {self.d:.2f} = 0"

def main():
    max_number_of_planes=3
    point_cloud_original = o3d.io.read_point_cloud("./data/factory_without_ground.ply")
    colormap=mpl.set1(np.linspace(0,1,max_number_of_planes))	
    point_cloud=deepcopy(point_cloud_original)
    planes=[]
    while True:
        plane=PlaneDetection(point_cloud)
        outlier_cloud=plane.segment()
        idx_color=len(planes)
        color=colormap[idx_color,0:3]
        plane.colorize_inliers(r=color[0],g=color[1],b=[2])

        planes.append(plane)

        if len(planes)>= max_number_of_planes:
            break
	
	#Create a list of entities to draw
    entities=[x.inlier_cloud for x in planes]
    entities.append(point_cloud)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])


if __name__ == '__main__':
    main()