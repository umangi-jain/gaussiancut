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

import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from arguments import ModelParams, PipelineParams
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2
import cv2
import numpy as np
import copy
from utils.sh_utils import RGB2SH


def render_video(model_path, iteration, views, gaussians, pipeline, background,
                 identifier):
    """
    Renders a video using the given parameters.

    Args:
        model_path (str): The path to the model.
        iteration (int): The iteration number.
        views (list): A list of views.
        gaussians (list): Optimized gaussians.
        pipeline (object): The pipeline object used in 3dgs module.
        background (object): The background color (usually white).
        identifier (str): The identifier for the video (object in scene usually).

    Returns:
        None
    """
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    render_poses = views

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(
        os.path.join(render_path, 'coarse_video_{}.mp4'.format(identifier)),
        fourcc, 10, size)

    for _, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(
            getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans,
                           view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(
            view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        rendering = torch.clamp(
            render(view, gaussians, pipeline, background)["render"], min=0.,
            max=1.)

        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() *
                           255.).astype(np.uint8)[..., ::-1])
    final_video.release()


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, image_name=None, override_colors=None):
    """
    Renders a set of images using the specified parameters.

    Args:
        model_path (str): The path to the model.
        name (str): The name of the set.
        iteration (int): The iteration number.
        views (list): A list of views.
        gaussians (list): A list of gaussians.
        pipeline (str): The pipeline to use.
        background (str): The background to use.
        image_name (str, optional): The name of the image. Defaults to None.
        override_colors (list, optional): A list of override colors. Defaults to None.
    """

    render_path = os.path.join(model_path, name, "renders")
    gts_path = os.path.join(model_path, name, "gt")
    if override_colors is not None:
        mask_path = os.path.join(model_path, name, "masks")
        makedirs(mask_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]

        gt = view.original_image[0:3, :, :]
        image_name = view.image_name if image_name else '{0:05d}'.format(idx)

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, image_name + ".png"))
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, image_name + ".png"))
        if override_colors is not None:
            rendering_mask = torch.clamp(
                render(view, gaussians, pipeline, background,
                       override_color=override_colors)["render"], min=0.,
                max=1.)
            torchvision.utils.save_image(
                rendering_mask, os.path.join(mask_path, image_name + ".png"))


def render_gc_sets(gaussians, scene, dataset: ModelParams, 
                   pipeline: PipelineParams,  skip_test: bool,
                   remove_gauss, identifier: str = "",
                   select_images: list = [], bkg_color='black'):
    """
    Finer Splatting.

    Args:
        gaussians (Gaussians): The Gaussian optimized for the scene.
        scene (Scene): The scene object used in 3dgs pipeline.
        dataset (ModelParams): The parameters of the model used in the 3dgs pipeline.
        pipeline (PipelineParams): The parameters of the rendering pipeline.
        skip_test (bool): Whether to skip rendering on test cameras.
        remove_gauss: The index indicating which Gaussians to remove.
        identifier (str, optional): An identifier for the rendered images. 
        select_images (list, optional): A list of image names to select for rendering. Defaults to [].
        bkg_color (str, optional): The background color. Defaults to 'black'.
    """
    
    with torch.no_grad():
        black_background = torch.tensor([0, 0, 0], dtype=torch.float32,
                                        device="cuda")
        white_background = torch.tensor([1, 1, 1], dtype=torch.float32,
                                        device="cuda")
        if bkg_color == 'black':
            background = black_background
        else:
            background = white_background
        override_colors = torch.zeros((gaussians._xyz.shape[0], 3),
                                      device="cuda")
        # wherever  remove_gauss is 1, we keep them with color [0, 0, 0]. Otherwise [1, 1, 1]
        override_colors[remove_gauss == 0] = white_background
        gaussians_colored = copy.deepcopy(gaussians)
        gaussians_colored._features_dc[remove_gauss == 1] = torch.ones(
            (int(remove_gauss.sum()), 1, 3), device="cuda") * RGB2SH(0)
        gaussians_colored._features_rest[remove_gauss == 1] = torch.zeros(
            (int(remove_gauss.sum()), 15, 3),
            device="cuda")  # gaussians._features_rest * 0.0
        if select_images:
            camera_list = scene.getTrainCameras().copy() + scene.getTestCameras(
            ).copy()
            camera_selected = []
            for camera_it in camera_list:
                if 'all' in select_images or camera_it.image_name in select_images:
                    camera_selected.append(camera_it)

            render_set(dataset.model_path,
                       "fine_gc_{}/select".format(identifier),
                       scene.loaded_iter, camera_selected, gaussians_colored,
                       pipeline, background, image_name=True,
                       override_colors=override_colors)


def render_coarse_sets(gaussians, scene, scene_path:str,dataset: ModelParams,
                       pipeline: PipelineParams,
                       mask_type: str,
                       foreground_threshold: float,
                       identifier: str = "", select_images: list = []):
    """
    Coarse splatting.

    Args:
        gaussians (Gaussians): The gaussians optimized for the scene.
        scene (Scene): The scene object used in 3dgs pipeline.
        scene_path (str): The path to the scene images.
        dataset (ModelParams): The parameters of the model.
        pipeline (PipelineParams): The parameters of the pipeline.
        mask_type (str): The type of mask to be used.
        foreground_threshold (float): The threshold for foreground selection.
        identifier (str, optional): An identifier for the render. Defaults to "".
        select_images (list, optional): A list of selected images. Defaults to [].
    """
    with torch.no_grad():

        proccessed_images = 0
        weights = torch.zeros_like(gaussians._opacity)
        weights_cnt = torch.zeros_like(gaussians._opacity)

        gaussians_for_sink = copy.deepcopy(gaussians)
        gaussians_for_source = copy.deepcopy(gaussians)
        weights_sink = torch.zeros_like(gaussians._opacity)
        weights_cnt_sink = torch.zeros_like(gaussians._opacity)

        if mask_type == 'spiral':
            interpolate_viewpoint = scene.getSpiralCameras().copy()
            mask_base_path = os.path.join(scene_path, "multiview_masks")
            masks_all = sorted(os.listdir(mask_base_path))
            assert len(masks_all) == len(
                interpolate_viewpoint
            ), "Number of masks and views do not match for all"
            for index, _ in enumerate(
                    tqdm(interpolate_viewpoint,
                         desc="Coarse rasterization progress")):
                mask_path = os.path.join(mask_base_path, masks_all[index])
                mask_image = Image.open(mask_path)
                transform = transforms.ToTensor()
                mask_tensor = transform(mask_image)
                _, height, width = interpolate_viewpoint[
                    index].original_image.shape
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).float(), size=(height, width),
                    mode="bilinear", align_corners=False).squeeze(0)
                mask = mask_tensor.repeat(3, 1, 1).contiguous().cuda()
                mask[mask > 0] = 1.0
                gaussians_for_source.apply_weights(interpolate_viewpoint[index],
                                                   weights, weights_cnt, mask)
                gaussians_for_sink.apply_weights(interpolate_viewpoint[index],
                                                 weights_sink, weights_cnt_sink,
                                                 1 - mask)
            print("Processed images (spiral): ", proccessed_images)
        elif mask_type == 'multiview':
            mask_base_path = os.path.join(scene_path, "multiview_masks")
            masks_all = sorted(os.listdir(mask_base_path))

            all_viewpoints = scene.getTrainCameras().copy() + \
                scene.getTestCameras().copy()

            images_names = []
            for index in range(len(masks_all)):
                image_name = masks_all[index].split('.')[0]
                images_names.append(image_name)
            select_viewpoints = [[] for _ in range(len(images_names))]

            for index in range(len(all_viewpoints)):
                for index_image, image_name in enumerate(images_names):
                    if all_viewpoints[index].image_name == image_name:
                        select_viewpoints[index_image] = all_viewpoints[index]
                        break
                    
            assert len(masks_all) == len(
                select_viewpoints
            ), "Number of masks and viewpoints do not match for all. Please check mask names!"
            for index, _ in enumerate(
                    tqdm(select_viewpoints,
                         desc="Coarse rasterization progress")):
                # skip rasterization for test cameras
                if select_viewpoints[index] in scene.getTestCameras():
                    continue
                mask_path = os.path.join(mask_base_path, masks_all[index])
                mask_image = Image.open(mask_path)
                transform = transforms.ToTensor()
                mask_tensor = transform(mask_image)

                _, height, width = select_viewpoints[index].original_image.shape
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).float(), size=(height, width),
                    mode="bilinear", align_corners=False).squeeze(0)
                mask = mask_tensor.repeat(3, 1, 1).contiguous().cuda()
                mask[mask > 0] = 1.0

                gaussians_for_source.apply_weights(select_viewpoints[index],
                                                   weights, weights_cnt, mask)
                gaussians_for_sink.apply_weights(select_viewpoints[index],
                                                 weights_sink, weights_cnt_sink,
                                                 1 - mask)
                proccessed_images += 1
            print("Processed images (multiview): ", proccessed_images)   

        elif mask_type == 'scribble':
            # this is used for the NVOS experiment.
            all_viewpoints = scene.getTrainCameras().copy() + \
                scene.getTestCameras().copy()
            mask_base_path = os.path.join(scene_path, "scribbles")
            masks_all = os.listdir(mask_base_path)
            image_name = masks_all[0].split('.')[0]
            for viewpoint_cam in range(len(all_viewpoints)):
                
                if viewpoint_cam.image_name == image_name:
                    proccessed_images += 1
                    mask_path = os.path.join(
                        mask_base_path,
                        viewpoint_cam.image_name + '_{}.png'.format(mask_type))

                    mask_image = Image.open(mask_path)
                    transform = transforms.ToTensor()
                    mask_tensor = transform(mask_image)
                    _, height, width = viewpoint_cam.original_image.shape
                    mask = F.interpolate(
                        mask_tensor.unsqueeze(0).float(), size=(height, width),
                        mode="bilinear", align_corners=False).squeeze(0)
                    mask[mask > 0] = 1.0
                    mask = mask.repeat(3, 1, 1).contiguous().cuda()

                    gaussians_for_source.apply_weights(viewpoint_cam, weights,
                                                       weights_cnt, mask)

                    mask_path_bk = os.path.join(
                        dataset.source_path, "mask_nvos",
                        viewpoint_cam.image_name +
                        '_{}_bk.png'.format(mask_type))
                    mask_image_bk = Image.open(mask_path_bk)
                    transform = transforms.ToTensor()
                    mask_tensor_bk = transform(mask_image_bk)
                    _, height, width = viewpoint_cam.original_image.shape
                    mask_bk = F.interpolate(
                        mask_tensor_bk.unsqueeze(0).float(),
                        size=(height, width), mode="bilinear",
                        align_corners=False).squeeze(0)

                    mask_bk[mask_bk > 0] = 1.0
                    mask_bk = mask_bk.repeat(3, 1, 1).contiguous().cuda()
                    gaussians_for_sink.apply_weights(viewpoint_cam,
                                                     weights_sink,
                                                     weights_cnt_sink, mask_bk)
                    break
            print("Processed images (scribble): ", proccessed_images)
        else:
            raise ValueError("Invalid mask type")
        weights = torch.where(weights_cnt == 0, torch.zeros_like(weights),
                              weights / weights_cnt)
        selected_mask = weights >= foreground_threshold

        print("Number of gaussians removed: ",
              torch.sum(selected_mask.int() == 0).item())
        print("Number of gaussians kept: ",
              torch.sum(selected_mask.int() == 1).item())
        print("Number of gaussians before: ", len(gaussians._opacity))

        gaussians_for_source.remove_low_score_gaussians(
            selected_mask.bool().squeeze(1))

        gaussians_for_source.save_ply(
            os.path.join(dataset.model_path, "graphcut_{}".format(identifier),
                         "gaussians_source_{}.ply".format(mask_type)))

        weights_sink = torch.where(weights_cnt_sink == 0,
                                   torch.ones_like(weights_sink),
                                   weights_sink / weights_cnt_sink)

        selected_mask_sink = weights < foreground_threshold
        print("Number of gaussians removed sink: ",
              torch.sum(selected_mask_sink.int() == 0).item())
        print("Number of gaussians kept sink: ",
              torch.sum(selected_mask_sink.int() == 1).item())
        gaussians_for_sink.remove_low_score_gaussians(
            selected_mask_sink.bool().squeeze(1))
        gaussians_for_sink.save_ply(
            os.path.join(dataset.model_path, "graphcut_{}".format(identifier),
                         "gaussians_sink_{}.ply".format(mask_type)))
        torch.save(
            selected_mask_sink,
            os.path.join(dataset.model_path, "graphcut_{}".format(identifier),
                         "weights_sink_{}.pt".format(mask_type)))
        torch.save(
            selected_mask,
            os.path.join(dataset.model_path, "graphcut_{}".format(identifier),
                         "weights_source_{}.pt".format(mask_type)))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if select_images:
            camera_list = scene.getTrainCameras().copy() + scene.getTestCameras(
            ).copy()
            camera_selected = []
            for camera_it in camera_list:
                if 'all' in select_images or camera_it.image_name in select_images:
                    camera_selected.append(camera_it)
            override_colors = torch.zeros((gaussians._xyz.shape[0], 3),
                                          device="cuda")
            selected_mask = selected_mask.squeeze()
            selected_mask = ~selected_mask
            override_colors[selected_mask == 0] = torch.tensor(
                [1, 1, 1], dtype=torch.float32, device="cuda")
            gaussians_colored = copy.deepcopy(gaussians)
            gaussians_colored._features_dc[selected_mask == 1] = torch.ones(
                (int(selected_mask.sum()), 1, 3), device="cuda") * RGB2SH(0)
            gaussians_colored._features_rest[selected_mask == 1] = torch.zeros(
                (int(selected_mask.sum()), 15, 3),
                device="cuda")  # gaussians._features_rest * 0.0
            render_set(dataset.model_path,
                       "coarse_{}/select".format(identifier), scene.loaded_iter,
                       camera_selected, gaussians_colored, pipeline, background,
                       image_name=True, override_colors=override_colors)
    return weights, weights_sink, gaussians_for_source, gaussians_for_sink
