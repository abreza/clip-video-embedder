import torch
import json
import random
import os
from tqdm import tqdm
import numpy as np
from metrics import compute_metrics
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from utils.utils import trim_sentences

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("/media/external_10TB/10TB/p_haghighi/Server71Backup/clip-model").to(device).eval()
processor = CLIPProcessor.from_pretrained('/media/external_10TB/10TB/p_haghighi/clip-processor')

print(f'The model successfully loaded on {device}.')

split = 'val1' #'train'
descriptions_file = f'/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/{split}.json'
descripiton_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/description/{split}"

with open(descriptions_file, 'r') as f:
    data = json.load(f)
    video_ids = data.keys()
def get_text_embedding(sentences):
    max_seq_len = 75
    while True:
        try:
            sentences = trim_sentences([" ".join(sentences)], max_seq_len=max_seq_len)
            text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_embedding = model.get_text_features(**text_inputs).float()
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            return text_embedding.view(-1)
        except RuntimeError as e:
            if "must match the size of tensor" in str(e):
                max_seq_len -= 1
                continue

for video_id in tqdm(video_ids,  desc='Calculating and saving text embeddings'):
    output_path = os.path.join(descripiton_embeddings_dir, video_id + ".pt")
    
    if os.path.exists(output_path): continue

    sentences = data[video_id]["sentences"]
    text_embedding = get_text_embedding(sentences)
    
    torch.save(text_embedding, output_path)