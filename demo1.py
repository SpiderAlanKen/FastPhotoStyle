"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

import process_stylization
from photo_wct import PhotoWCT
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_image_path', default='./images/content1.png')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default='./images/content3.png')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/example1.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=True)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
args = parser.parse_args()

# 遍历文件夹
dir_content = r'C:\Users\10929\PycharmProjects\FastPhotoStyle\images\content'
dir_style = r'C:\Users\10929\PycharmProjects\FastPhotoStyle\images\style'
search_scope = r'*.png'

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))
if args.cuda:
    p_wct.cuda(0)

for file_content in tqdm(Path(dir_content).rglob(search_scope)):
    for file_style in Path(dir_style).rglob(search_scope):
        args.content_image_path = os.fspath(file_content)
        args.style_image_path = os.fspath(file_style)
        # frn = file_style.stem + '_'+file_content.name # 重命名
        frn = file_style.stem +'_'+file_content.stem + '_第二遍.png'  # 重命名
        args.output_image_path = os.fspath(Path(r'C:\Users\10929\PycharmProjects\FastPhotoStyle\results').joinpath(frn))

        if args.fast:
            from photo_gif import GIFSmoothing
            p_pro = GIFSmoothing(r=35, eps=0.001)
        else:
            from photo_smooth import Propagator
            p_pro = Propagator()

        process_stylization.stylization(
            stylization_module=p_wct,
            smoothing_module=p_pro,
            content_image_path=args.content_image_path,
            style_image_path=args.style_image_path,
            content_seg_path=args.content_seg_path,
            style_seg_path=args.style_seg_path,
            output_image_path=args.output_image_path,
            cuda=args.cuda,
            save_intermediate=args.save_intermediate,
            no_post=args.no_post
        )



