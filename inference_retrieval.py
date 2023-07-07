import torch
import json
import random
import os
from tqdm import tqdm
import numpy as np
from metrics import compute_metrics
import torch.nn as nn
# from transformers import CLIPModel, CLIPProcessor
import argparse
from diverse_frame_selector import select_diverse_frames

parser = argparse.ArgumentParser(description='Different methods of frame selection')
parser.add_argument('--mode', type=str, help='mode string', default='top')
parser.add_argument('-k', type=int, help='integer value of k', default=8)
args = parser.parse_args()

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# model = CLIPModel.from_pretrained("/media/external_10TB/10TB/p_haghighi/Server71Backup/clip-model").to(device).eval()
# processor = CLIPProcessor.from_pretrained('/media/external_10TB/10TB/p_haghighi/clip-processor')
# print(f'The model successfully loaded on {device}.')

split = 'val_1'
frames_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/frames/{split}"
descripiton_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/description/{split}"
saliency_json_file = f"/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/captions_gt_{split}.json"

with open(saliency_json_file, "r") as f:
    saliency_dict = json.load(f)

def get_video_embeddings(batch_video_ids, mode='k-top', number_of_frames=8):
    
    embeddings = []
    for video_id in batch_video_ids:

        frames_embedding = torch.load(os.path.join(frames_embeddings_dir, video_id + ".pt"), map_location=device)

        k = min(frames_embedding.size(0), number_of_frames)

        if mode == 'top':
            top_indices = sorted(range(len(saliency_dict[video_id])), key=lambda i: saliency_dict[video_id][i], reverse=True)[:k]
        elif mode == 'top+diverse':
            top_indices = select_diverse_frames(video_id, k)
        elif mode == 'random':
            top_indices = random.sample(range(len(frames_embedding)), k)
        elif mode == 'uniform':
            step = len(frames_embedding) // k
            top_indices = list(range(0, len(frames_embedding), step))[:k]
        elif mode == 'least':
            top_indices = sorted(range(len(saliency_dict[video_id])), key=lambda i: saliency_dict[video_id][i])[:k]
        else:
            raise ValueError(f"Invalid mode. Mode must be one of 'top', 'least', 'random', 'uniform', 'top+diverse'.")
        
        selected_frames_embedding = frames_embedding[top_indices]
        video_embedding = selected_frames_embedding.mean(dim=0)
        
        embeddings.append(video_embedding)

    return torch.stack(embeddings, dim=0)

def get_text_embeddings(batch_video_ids):
    embeddings = []
    for video_id in batch_video_ids:
        text_embedding = torch.load(os.path.join(descripiton_embeddings_dir, video_id + ".pt"), map_location=device)
        embeddings.append(text_embedding)
    return torch.stack(embeddings, dim=0)

batch_text_features_list, batch_video_features_list = [], []

video_ids = list(saliency_dict.keys())  # get list of video ids
batch_size = 64  # set batch size
batch_video_features_list = []
batch_text_features_list = []

num_batches = (len(video_ids) + batch_size - 1) // batch_size  # calculate number of batches

# for i in tqdm(range(num_batches), desc='Cache the features'):
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(video_ids))
    batch_video_ids = video_ids[start_idx:end_idx]  # get a batch of video ids
    video_features = get_video_embeddings(batch_video_ids, mode = args.mode, number_of_frames = args.k)
    text_features  = get_text_embeddings(batch_video_ids)

    batch_text_features_list.append(text_features)
    batch_video_features_list.append(video_features)


sim_matrix = []

# for text_features in tqdm(batch_text_features_list, desc='Calculate the similarity'):
for text_features in batch_text_features_list:
    each_row = []
    
    for video_features in batch_video_features_list:

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = nn.Parameter(torch.ones([])).exp()
        b1b2_logits = torch.matmul(text_features, video_features.t()) * logit_scale
        b1b2_logits = b1b2_logits.cpu().detach().numpy()
        each_row.append(b1b2_logits)

    each_row = np.concatenate(tuple(each_row), axis=-1)
    sim_matrix.append(each_row)


sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

tv_metrics = compute_metrics(sim_matrix)
# print("Text-to-Video:")
print('k:{}\tR@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
            format(args.k, tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

vt_metrics = compute_metrics(sim_matrix.T)
# print("Video-to-Text:")
print('k:{}\tR@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
            format(args.k, vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))