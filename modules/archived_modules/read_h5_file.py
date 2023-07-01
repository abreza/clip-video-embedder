import h5py
import json
import torch
import numpy as np

json_path = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/merged_file.json"
h5_path = "/media/external_10TB/10TB/p_haghighi/CLIP_guided_dataset/merged_output.h5"

all_features, all_gtscores = [], []


with open(json_path) as f:
    data = json.load(f)


with h5py.File(h5_path, 'r') as hdf:
    
    for video_name in data.keys():
        group = hdf.get(video_name)
        
        features = np.array(group.get('features'))
        gtscore = np.array(group.get('gtscore'))
        
        all_features.append(features)
        all_gtscores.append(gtscore)
        
        # You can then use the features and gtscore variables in your downstream task
        print("Video:", video_name)
        print("Features shape:", features.shape)
        print("Gtscore shape:", gtscore.shape)

        break