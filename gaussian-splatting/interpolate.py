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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2
import cv2
import numpy as np
import sys
from PIL import Image
import torch.nn.functional as F

def render_video(model_path, iteration, views, gaussians, pipeline, background, identifier, save_image=False): ###
    render_path = os.path.dirname(os.path.dirname(model_path))
    render_path = os.path.join(render_path, 'video_for_seg') 
    makedirs(render_path, exist_ok=True)
    view = views[0]
    if save_image:
        render_image_path = os.path.join(model_path, 'video_images' )
        makedirs(render_image_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, '{}.mp4'.format(identifier)), fourcc, 10, size)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        if save_image:
            torchvision.utils.save_image(rendering, os.path.join(render_image_path, '{0:05d}'.format(idx) + "_interpolate.png"))

        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, final_path : str = None, 
                identifier : str = "", scene_type : str = "frontfacing"):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")      
        render_video(dataset.model_path, scene.loaded_iter, scene.getSpiralCameras(), 
                     gaussians, pipeline, background, identifier)
        

def render_custom_video(model_path, identifier):
    check_custom_path = os.path.join(model_path, 'images_for_video')
    if os.path.exists(check_custom_path):
        check_custom_path = check_custom_path
        max_images = -1
    else:
        check_custom_path = os.path.join(model_path, 'images')
        identifier = identifier + '_rand' 
        max_images = 40
    video_path = os.path.join(model_path, 'video_for_seg')
    os.makedirs(video_path, exist_ok=True)
    video_name = os.path.join(video_path, '{}.mp4'.format(identifier))
    images = os.listdir(check_custom_path)
    images.sort()
    read_imags = [Image.open(os.path.join(check_custom_path, img)) for img in images]
    read_images = []
    image_shape = np.array(read_imags[0]).shape[:2]
    for img in read_imags:
        full_image = np.array(img)
        full_image = torch.from_numpy(full_image).permute(2, 0, 1).unsqueeze(0).float()/255
        full_image = F.interpolate(full_image, size=image_shape, mode='bilinear', align_corners=False).squeeze().permute(1, 2, 0).numpy()
        full_image = (full_image*255).astype(np.uint8)
        read_images.append(full_image)
        if len(read_images) == max_images:
            break
    read_images = np.stack([img for img in read_images])
    print(read_images.shape)
    size = (read_images.shape[2], read_images.shape[1])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or you can use 'XVID'
    video = cv2.VideoWriter(video_name, fourcc, 5, size)  # 1 FPS for demonstration
    for img in read_images:
        video.write(img.astype(np.uint8)[..., ::-1])

    video.release()
    return None

        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--final_path", default='', type=str)
    parser.add_argument("--identifier", default='', type=str)
    # frontfacing, custom
    
    parser.add_argument("--scene_type", default='frontfacing', type=str)
    args = parser.parse_args(sys.argv[1:])
    # if scene type is frontfacing, then interpolate the view on the spiral path
    if args.scene_type == 'frontfacing':
        args = get_combined_args(parser)
        print("Rendering " + args.model_path)
        safe_state(args.quiet)
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, 
                    args.skip_test, args.final_path, args.identifier)
    elif args.scene_type == 'custom':
        # provide the set of images aranged in order to create the desired video
        render_custom_video(args.model_path, args.identifier)
    else:
        raise ValueError("Scene type not supported")