# Selects diverse frames using a modified Sieve of Eratosthenes algorithm.
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

split = 'val_1'
frames_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/frames/{split}"
descripiton_embeddings_dir = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/CLIP_embeddings/description/{split}"
saliency_json_file = f"/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/captions_gt_{split}.json"
frames_path = f"/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/{split}"


with open(saliency_json_file, "r") as f:
    saliency_dict = json.load(f)

def select_diverse_frames(video_id, k, similarity_threshold=0.9):
    
    saliencies = saliency_dict[video_id]
    sorted_indices = np.argsort(saliencies)[::-1]  # Sort frames by saliency in descending order

    return sorted_indices[:k]

    embeddings = torch.load(os.path.join(frames_embeddings_dir, video_id + ".pt"), map_location='cpu')
    sorted_embeddings = [(i, embeddings[i]) for i in sorted_indices]

    selected_indices = []
    for i in range(sorted_indices.shape[0]):
        if len(selected_indices) >= k: 
            break

        current_embedding = sorted_embeddings[i][1]
        is_similar = False

        for j in selected_indices:
            similarity = cosine_similarity([current_embedding.detach().numpy()], [sorted_embeddings[j][1].detach().numpy()])[0][0]
            if similarity > similarity_threshold:
                is_similar = True
                break

        if not is_similar:
            selected_indices.append(sorted_embeddings[i][0])

    return selected_indices


video_ids = list(saliency_dict.keys())
video_id = 'v_uM3RiCL0g2U'
k = 8

selected_indices = select_diverse_frames(video_id, k=k, similarity_threshold=0.9)
print(f'selected indices: {selected_indices}')

# Create a figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=k//2, figsize=(12, 6))

# Loop over the selected indices and plot the corresponding frames
for i, idx in enumerate(selected_indices):

    frame_file = os.path.join(frames_path, video_id, f'{video_id}_frame{idx:03d}.jpg')

    img = mpimg.imread(frame_file)

    row, col = i//4, i%4

    axs[row, col].imshow(img)
    axs[row, col].set_title(f'Frame {idx}')

    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])

# Adjust the spacing between subplots
fig.tight_layout()
fig.savefig('selected_frames.png')
plt.close(fig)