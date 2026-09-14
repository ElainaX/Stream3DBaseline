import numpy as np
import open_clip
from open_clip import tokenizer
import torch
from evaluation.constants import MATTERPORT_LABELS, SCANNET_LABELS, SCANNETPP_LABELS
import warnings
from utils.config import get_dataset, get_args
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def load_clip():
    print(f'[INFO] loading CLIP model...')
    # model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="./weights/open_clip_model.safetensors")
    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model

def extract_text_feature(save_path, descriptions, model):
    text_tokens = tokenizer.tokenize(descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    text_features_dict = {}
    for i, description in enumerate(descriptions):
        text_features_dict[description] = text_features[i]

    # print(save_path)
    np.save(save_path, text_features_dict)

def main(args):
    model = load_clip()
    if args.dataset == 'scannet':
        print('data/text_features/scannet.npy')
        extract_text_feature('data/text_features/scannet.npy', SCANNET_LABELS, model)
    if args.dataset == 'scannetpp':
        print('data/text_features/scannetpp.npy')
        extract_text_feature('data/text_features/scannetpp.npy', SCANNETPP_LABELS, model)
    if args.dataset == 'matterport3d':
        print('data/text_features/matterport3d.npy')
        extract_text_feature('data/text_features/matterport3d.npy', MATTERPORT_LABELS, model)

if __name__ == '__main__':
    args = get_args()
    main(args)