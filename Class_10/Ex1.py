#!/usr/bin/env python3

import copy
from pydoc import locate
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import matplotlib as cm
from class_plain_detection import PointCloudProcessing
#from colormap import Colormap as cm

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
			"lookat" : [ 0, 0,0 ],
			"up" : [ -0.20703638500269855, 0.25076680911690946, 0.9456489532222504 ],
			"zoom" : 0.3412
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def main():
    p=PointCloudProcessing()
    p.loadPointCloud('./data/scene.ply')
    source = o3d.io.read_point_cloud('data/cereal_box_2_2_40.pcd')
    p.preProcess()
    p.transform(-108,0,0,0,0,0)
    p.transform(0,0,-37,0,0,0)
    p.transform(0,0,0,0,0,.5)

    p.crop(-0.9,-0.9,0,0.9,0.9,0.9)

    outliers= p.findPlane()
    p.inliers.paint_uniform_color([0,1,1])

    cluster_idxs = np.array(outliers.cluster_dbscan(eps=0.05, min_points=50, print_progress=True))
    object_idxs=list(set(cluster_idxs))
    object_idxs.remove(-1)

    objects = []
    number_of_objects=len(object_idxs)
    #colormap=cm.Pastel1(list(range(0,number_of_objects)))
    for object_idx in object_idxs:
        object_point_idxs=list(locate(cluster_idxs,lambda x: x==object_idx))
        object_points=outliers.select_by_index(object_point_idxs)
        d={}
        d['idx']=str(object_idx)
        d['points']=object_points
        #d['color']=colormap[object_idx,0:3]
        #color = d['color']
        color=[1,1,1]
        d['points'].paint_uniform_color(color)
        d['center']=d['points'].get_center()
        objects.append(d)    
    
    entities=[]
    for idx,object in enumerate(objects):
        #if idx==2:
        entities.append(object['points'])
        print("Apply point-to-point ICP")
        trans_init = np.asarray([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0], 
                                 [0, 0, 0, 1]])
        reg_p2p = o3d.pipelines.registration.registration_icp(source, object['points'], 0.3, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
        object['inlier_rmse']=reg_p2p.inlier_rmse
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(source, object['points'], reg_p2p.transformation)
    
    minimum_rmse=10e8
    cereal_box_object_idx=None
    for object,idx in enumerate(objects):
        if object['inlier_rmse'] < minimum_rmse:
            minimum_rmse=object['inlier_rmse']
            cereal_box_object_idx=idx

    
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5,origin=np.array([0.,0.,0.]))
    

 
    entities.append(frame)
    entities.append(p.pcd)
    bbox_to_draw=o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox) 
    entities.append(bbox_to_draw)
    entities.append(outliers)
    # o3d.visualization.draw_geometries(entities,
    #                                 zoom=view['trajectory'][0]['zoom'],
    #                                 front=view['trajectory'][0]['front'],
    #                                 lookat=view['trajectory'][0]['lookat'],
    #                                 up=view['trajectory'][0]['up'])

    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 5 * w.scaling

    for entity_idx,entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity" + str(entity_idx), entity, material)
        
    for object_idx,object in enumerate(objects):    
        label_pos = [object['center'][0],object['center'][1],object['center'][2]+0.2]
        l = widget3d.add_3d_label(label_pos, object['idx'])
        l.color = gui.Color(object['color'], object['color'], object['color'])
        l.scale = 3
    
   

    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()


if __name__ == '__main__':
    main()