import json
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


saliency_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/by labels/gpt_prompt_gt_val1.json"
sampled_frames_dir = "/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/val_1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18(pretrained=True).eval().to(device)

# define the transformations to apply to the frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open(saliency_file, "r") as f:
    saliency_dict = json.load(f)

for video_id, scores in saliency_dict.items():
    # sort the scores in descending order and get the top 8 indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
    # get the corresponding frame names
    top_frames = [f"v_QoTM5tmcJeI_frame{i+1:03}.jpg" for i in top_indices]
    # construct the path to the video's frames directory
    frames_dir = os.path.join(sampled_frames_dir, video_id)
    # construct the paths to the top frames
    top_frame_paths = [os.path.join(frames_dir, frame_name) for frame_name in top_frames]
    # pass the top frames to the ResNet-18
    # and use the output to decide the class of the video
    _ = [print(t) for t in top_frame_paths]
       


    # pass each frame through the model and store the logits
    logits = []
    for frame_path in top_frame_paths:
        with open(frame_path, "rb") as f:
            frame = Image.open(f).convert("RGB")
            frame = transform(frame)
            frame = frame.unsqueeze(0)
            # pass the frame through the ResNet-18 model
            output = resnet18(frame)
            logits.append(output)

    # average the logits to obtain the final prediction
    logits = torch.cat(logits, dim=0)
    prediction = logits.mean(dim=0)

    print(prediction)

    break