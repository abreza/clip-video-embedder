import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("./clip-model").to(device).eval()
processor = CLIPProcessor.from_pretrained("./clip-processor")

video_dir = "/media/10TB71/ramezani/ACTIVITY_NET/CLIP_embeddings/pt_files"
json_file = "/home/ramezani/haghighi/clip-video-embedder/datasets/ActivityNet/merged_file.json"

def trim_sentences(sentences, max_seq_len=75):
    trimmed_sentences = []
    for sentence in sentences:
        tokens = sentence.strip().split()[:max_seq_len]
        trimmed_sentence = " ".join(tokens)
        trimmed_sentences.append(trimmed_sentence)
    return trimmed_sentences

def inference(model, processor, sentences : list, image_features : list):
    torch.cuda.empty_cache()

    # Check if sentences is empty or contains an empty string
    if not sentences or "" in sentences:
        return []

    text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).float()
    text_features  /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity = np.max(similarity, axis=0)
    similarity = np.round(similarity, 3)
    return similarity.tolist()


# load the video IDs from the JSON file
with open(json_file) as f:
    data = json.load(f)

video_ids = set(data.keys())

outputs = {}
with tqdm(total=len(video_ids)) as pbar:
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".pt"):
                video_id = file.replace('.pt','')
                if video_id not in video_ids:
                    continue
                image_features = torch.load(os.path.join(root, file))
                sentences = data[video_id] 
                outputs[video_id] = inference(model, processor, trim_sentences(sentences,max_seq_len=69), image_features)
                pbar.update(1)
            

print("Inference complete!")

# save outputs to a JSON file
with open('inference_outputs.json', 'w') as f:
    json.dump(outputs, f)