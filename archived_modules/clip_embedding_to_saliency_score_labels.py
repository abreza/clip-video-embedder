import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("/media/external_10TB/10TB/p_haghighi/Server71Backup/clip-model").to(device).eval()
processor = CLIPProcessor.from_pretrained('/media/external_10TB/10TB/p_haghighi/clip-processor')

print(f'the model successfully loaded on {device}.')

video_dir = '/media/external_10TB/10TB/p_haghighi/Server71Backup/CLIP_embeddings/'

label_json_file = '/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/activity_net.v1-3.min.json'
gpt_json_file = '/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/gpt_prompts.json'

# Load the JSON file
with open(label_json_file, 'r') as f:
    label_json_data = json.load(f)['database']

with open(gpt_json_file, 'r') as f:
    gpt_json_data = json.load(f)


def generate_template_prompts(action_label):
    prompt_templates = [
        "This video shows someone performing a {}.",
        "A {} is being performed in this video.",
        "The action in this video is {}.",
        "This video depicts a {} being performed.",
        "In this video, we can see someone doing {}.",
        "A {} is taking place in this video.",
        "This video shows someone engaged in {}.",
        "The subject of this video is performing a {}.",
        "This video depicts a {} in progress.",
        "In this video, someone is {}."
    ]
    prompts = [template.format(action_label.lower()) for template in prompt_templates]
    return prompts

def generate_gpt_prompts(action_label):
    return gpt_json_data[action_label]

def inference(model, processor, sentences : list, image_features : list):
    torch.cuda.empty_cache()

    text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).float()
    text_features  /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity = np.max(similarity, axis=0)
    similarity = np.round(similarity, 3).tolist()
    return similarity

def get_similarity(use_gpt_prompts):
    output_json = '/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/'
    output_json += 'gpt_prompt_gt_' if use_gpt_prompts else 'template_prompt_gt_'
        
    for mode in ['train', 'val1']:

        pt_files = os.listdir(os.path.join(video_dir, mode))

        outputs = {}

        print_prompt = 'gpt_prompt' if use_gpt_prompts else 'template_prompt'
        print(f"Inference of {mode} with {print_prompt}")

        for pt_file in tqdm(pt_files, desc='Processing video files'):

            video_id = pt_file.replace('.pt','')

            image_features_file_path = os.path.join(video_dir, mode, pt_file)
            image_features = torch.load(image_features_file_path)
            video_label = label_json_data[video_id[2:]]['annotations'][0]['label']

            prompts = generate_gpt_prompts(video_label) if use_gpt_prompts else generate_template_prompts(video_label)

            outputs[video_id] = inference(model, processor, prompts, image_features)
                
                    
        # save outputs to a JSON file
        with open(output_json + mode + '.json', 'w') as f:
            json.dump(outputs, f, indent=2)


# get_similarity(False) # has already fone. don't uncomment it!
get_similarity(True)

# I cannot find out any paper that use a specical kind of prompt matching:

# AAAI'2023: Revisiting Classifier: Transferring Vision-Language Models for Video Recognition : A video of a person {}
# CVPR'2023:🚴 BIKE: Bidirectional Cross-Modal Knowledge Exploration for Video Recognition with Pre-trained Vision-Language Models : This is a video about {}
# ECCV'2022: Zero-Shot Temporal Action Detection via Vision-Language Prompting : A video of {}
