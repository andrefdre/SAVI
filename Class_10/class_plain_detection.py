#!/usr/bin/env python3

from copy import deepcopy
from pydoc import locate
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class PointCloudProcessing():
    def loadPointCloud(self,filename):
        self.pcd = o3d.io.read_point_cloud(filename)
        self.original=deepcopy(self.pcd)

    def preProcess(self):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,max_nn=30))
        self.pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,0.]))

    def transform(self,r,p,yaw,x,y,z):
        rotation=self.pcd.get_rotation_matrix_from_xyz((r,p,yaw))
        self.pcd.rotate(rotation,center=(0,0,0))
        self.pcd=self.pcd.translate((x,y,z))

    def crop(self,min_x,min_y,min_z,max_x,max_y,max_z):
        np_points=np.ndarray((8,3),dtype=float)
        np_points[0,:]=[min_x,min_y,min_z]
        np_points[1,:]=[max_x,min_y,min_z]
        np_points[2,:]=[max_x,max_y,min_z]
        np_points[3,:]=[max_x,max_y,min_z]

        np_points[4,:]=[min_x,min_y,max_z]
        np_points[5,:]=[max_x,min_y,max_z]
        np_points[6,:]=[max_x,max_y,max_z]
        np_points[7,:]=[max_x,max_y,max_z]

        print(np_points)
        
        
        bbox_points=o3d.utility.Vector3dVector(np_points)
        self.bbox= o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)
        self.pcd=self.pcd.crop(self.bbox)

    def findPlane(self,distance_threshold=0.025,ransac_n=3,num_iterations=100):
        plane_model, inliers_idx = self.pcd.segment_plane(distance_threshold=distance_threshold,ransac_n=ransac_n,num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model
        self.inliers = self.pcd.select_by_index(inliers_idx)
        outlier_cloud = self.pcd.select_by_index(inliers_idx, invert=True)
        return outlier_cloud