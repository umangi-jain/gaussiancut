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
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, GraphCutParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphcut import load_gc_args, graphcut_segmentation
from utils.render_utils import render_gc_sets, render_coarse_sets
import copy


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    graphcutparams = GraphCutParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # could be spiral (used when masks are generated from a trajectory), 
    # multiview (with unordered images and masks), 
    # or scribble (when position and negative scribbles are used)
    parser.add_argument("--mask_type", default='multiview', type=str)
    # threshold for foreground and background during coarse splatting (0.3 for 360 inward scene)
    parser.add_argument("--foreground_threshold", default=0.9, type=float)
    # identifier used to save all the weights/gaussians
    parser.add_argument("--identifier", default="", type=str)
    # used when only some selected images are to be rendered
    parser.add_argument("--select_images", nargs="+", type=str, default=[])
    # directory where the data is stored
    parser.add_argument("--scene_path", type=str, default='')
    # skip coarse splatting it its output are already saved
    parser.add_argument("--skip_coarse", default=False, type=bool)
    # skip graphcut if its output are already saved
    parser.add_argument("--skip_gc", default=False, type=bool)

    # add graphcut params
    load_gc_args(parser)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    if not args.skip_coarse:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration,
                      shuffle=False)

        weights_source, weights_sink, gs_source, gc_sink = render_coarse_sets(
            gaussians, scene, args.scene_path, dataset, pipeline.extract(args),
            args.mask_type,
            args.foreground_threshold, 
            args.identifier, args.select_images)
    if not args.skip_gc:
        # start graph cut.
        if args.skip_coarse:
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=args.iteration,
                          shuffle=False)
            weights_source = os.path.join(
                dataset.model_path, "graphcut_{}".format(args.identifier),
                "weights_source_{}.pt".format(args.mask_type))
            weights_sink = os.path.join(
                dataset.model_path, "graphcut_{}".format(args.identifier),
                "weights_sink_{}.pt".format(args.mask_type))
            weights_source = torch.load(weights_source).int()
            weights_sink = torch.load(weights_sink).int()
            gc_sink = copy.deepcopy(gaussians)
            gs_source = copy.deepcopy(gaussians)
            gs_source.remove_low_score_gaussians(
                weights_source.bool().squeeze(1))
            gc_sink.remove_low_score_gaussians(weights_sink.bool().squeeze(1))
        foreground, background_index = graphcut_segmentation(
            args, dataset.model_path, graphcutparams, weights_source,
            weights_sink, gaussians, gs_source, gc_sink)
        render_gc_sets(gaussians, scene, dataset,
                       pipeline.extract(args), args.skip_test,
                        background_index, args.identifier,
                       args.select_images, 'black')

