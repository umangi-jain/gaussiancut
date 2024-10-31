#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.camera_utils import Camera
import math

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    def getSpiralCameras(self, scale=1.0, camera_count=30, rots=1, zrate=0.5, T_range=0.5, 
                         angle_range=0.1):
        cameras = self.getTrainCameras(scale)
        angles = np.stack([R.from_matrix(cam.R).as_euler('zyx', degrees=True) for cam in cameras])
        Ts = np.stack([cam.T for cam in cameras])

        min_angle = np.min(angles, axis=0)
        max_angle = np.max(angles, axis=0)
        min_T = np.min(Ts, axis=0)
        max_T = np.max(Ts, axis=0)

        if T_range < 1:
            T_len = max_T - min_T
            min_T += T_len * (1 - T_range) / 2
            max_T -= T_len * (1 - T_range) / 2

        if angle_range < 1:
            angle_len = max_angle - min_angle
            min_angle += angle_len * (1 - angle_range) / 2
            max_angle -= angle_len * (1 - angle_range) / 2

        spiral_cameras = []
        for theta in np.linspace(0., 2. * np.pi * rots, camera_count + 1)[:-1]:
            cam = Camera(
                colmap_id=0,
                R=R.from_euler('zyx', self.angle_interpolation(min_angle, max_angle, theta), 
                               degrees=True).as_matrix(),
                T=self.position_interpolation(min_T, max_T, theta, zrate),
                FoVx=cameras[0].FoVx,
                FoVy=cameras[0].FoVy,
                image=cameras[0].original_image,
                gt_alpha_mask=None,
                image_name=f"spiral",
                uid=0,
            )
            spiral_cameras.append(cam)
        return spiral_cameras


    def angle_interpolation(self, min, max, t):
        tmp = (np.sin(t) + 1) / 2
        return min + (max - min) * tmp
    
    def position_interpolation(self, min, max, t, zrate):
        tmp = np.array([
            (np.sin(t) + 1) / 2,
            (np.cos(t) + 1) / 2,
            (-np.sin(t * zrate) + 1) / 2
        ])
        return min + (max - min) * tmp