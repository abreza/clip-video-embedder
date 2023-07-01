import torch
import json
import random
import os
from tqdm import tqdm
import numpy as np
from metrics import compute_metrics
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import argparse

# suppress all INFO and WARNING messages, and only display ERROR messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='Different methos of frame selection')

parser.add_argument('--mode', type=str, help='mode string')
parser.add_argument('-k', type=int, help='integer value of k')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = CLIPModel.from_pretrained("/media/external_10TB/10TB/p_haghighi/Server71Backup/clip-model").to(device).eval()
# processor = CLIPProcessor.from_pretrained('/media/external_10TB/10TB/p_haghighi/clip-processor')
# print(f'The model successfully loaded on {device}.')

split = 'val1'
frames_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/frames/{split}"
descripiton_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/description/{split}"
saliency_json_file = f"/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/label_gpt_prompt_gt_{split}.json"

with open(saliency_json_file, "r") as f:
    saliency_dict = json.load(f)

def get_video_embeddings(batch_video_ids, mode='k-top', k=8):
    
    embeddings = []
    for video_id in batch_video_ids:
        frames_embedding = torch.load(os.path.join(frames_embeddings_dir, video_id + ".pt"), map_location=device)
        k = min(len(frames_embedding), k)

        if mode == 'k-top':
            top_indices = sorted(range(len(saliency_dict[video_id])), key=lambda i: saliency_dict[video_id][i], reverse=True)[:k]
        elif mode == 'k-least':
            top_indices = sorted(range(len(saliency_dict[video_id])), key=lambda i: saliency_dict[video_id][i])[:k]
        elif mode == 'random':
            top_indices = random.sample(range(len(frames_embedding)), k)
        else:
            raise ValueError("Invalid mode. Mode must be one of 'k-top', 'k-least', or 'random'")
        
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

# ----------------------------- Code of CLIP4Clip -------------------------------------
#  https://github.com/ArrowLuo/CLIP4Clip/blob/master/main_task_retrieval.py#L321C1-L460

batch_text_features_list, batch_video_features_list = [], []

video_ids = list(saliency_dict.keys())  # get list of video ids
batch_size = 16  # set batch size
batch_video_features_list = []
batch_text_features_list = []

num_batches = (len(video_ids) + batch_size - 1) // batch_size  # calculate number of batches

# for i in tqdm(range(num_batches), desc='Cache the features'):
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(video_ids))
    batch_video_ids = video_ids[start_idx:end_idx]  # get a batch of video ids
    video_features = get_video_embeddings(batch_video_ids, mode = args.mode, k = args.k)
    text_features  = get_text_embeddings(batch_video_ids)

    batch_text_features_list.append(text_features)
    batch_video_features_list.append(video_features)


sim_matrix = []

# for text_features in tqdm(batch_video_features_list, desc='Calculate the similarity'):
for text_features in batch_video_features_list:
    each_row = []
    
    for video_features in batch_video_features_list:

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features.squeeze(1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = nn.Parameter(torch.ones([])).exp()
        b1b2_logits = torch.matmul(text_features, video_features.t()) #* logit_scale
        b1b2_logits = b1b2_logits.cpu().detach().numpy()
        each_row.append(b1b2_logits)

    each_row = np.concatenate(tuple(each_row), axis=-1)
    sim_matrix.append(each_row)


sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

def skip(): #in CLIP4Clip, this is body of if multi_sentence
# sentences_dict = {}
# cut_off_points = []
# for video_id in video_ids:
#     for cap in captions[video_id]:
#         cap_txt = " ".join(cap)
#         sentences_dict[len(sentences_dict)] = (video_id, cap_txt)
#     cut_off_points.append(len(sentences_dict))

# if multi_sentence_:
#     print("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
#     max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points[:-1], cut_off_points)])
#     sim_matrix_new = []
#     for s_, e_ in zip([0] + cut_off_points[:-1], cut_off_points):
#         sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
#                                                 np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
#     sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
#     print("after reshape, sim matrix size: {} x {} x {}".
#                 format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

#     tv_metrics = tensor_text_to_video_metrics(sim_matrix)
#     vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

# else:
    pass

#in the case of note multi_sntence, we have jsut these 2 lines:
#multi_sentence means having multi descriptions pair a video (like ActivityNet)
tv_metrics = compute_metrics(sim_matrix)
# vt_metrics = compute_metrics(sim_matrix.T)


# print("Text-to-Video:")
print('R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
            format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

# print("Video-to-Text:")
# print('V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
#             format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))