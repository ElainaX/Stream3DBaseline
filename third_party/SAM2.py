import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import multiprocessing as mp
import os
from tqdm import tqdm
import glob
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def get_parser():
    parser = argparse.ArgumentParser(description="SAM")
    parser.add_argument(
        "--seq_name_list",
        type=str
    )
    parser.add_argument(
        "--root",
        type=str
    )
    parser.add_argument(
        "--image_path_pattern",
        type=str
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    return parser


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    seq_name_list = args.seq_name_list.split('+')

    device = "cuda"

    sam2_checkpoint = "....../Stream3D/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    for i, seq_name in tqdm(enumerate(seq_name_list), total=len(seq_name_list)):
        seq_dir = os.path.join(args.root, seq_name)
        image_list = sorted(glob.glob(os.path.join(seq_dir, args.image_path_pattern)))
        output_dir = os.path.join(seq_dir, seq_name, 'output_SAM2/mask') if args.dataset == 'matterport3d' else os.path.join(seq_dir, 'output_SAM2/mask')
        os.makedirs(output_dir, exist_ok=True)

        i = 1

        path = image_list[0]
        image = cv2.imread(path)

        fixed_m_H, fixed_m_W = image.shape[0], image.shape[1]

        for path in (image_list):

            i += 1

            image = cv2.imread(path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks_l = mask_generator.generate(image)
            sorted_anns = sorted(masks_l, key=(lambda x: x['area']), reverse=True)

            if len(sorted_anns) == 0:
                mask_image = np.ones((fixed_m_H, fixed_m_W), dtype=np.uint8)
                cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'), mask_image)
                continue

            mask_image = np.zeros((fixed_m_H, fixed_m_W), dtype=np.uint8)
            mask_id = 1
            delta = int(250 / len(sorted_anns))
            
            for ann in sorted_anns:
                m = ann['segmentation'] * 1
                num_pixels = np.sum(m)
                if num_pixels < 400: # ignore small masks
                    continue
                mask_image[(m==1)] = mask_id
                mask_id += delta

            cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'), mask_image)
