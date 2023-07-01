import os
import time
import numpy as np
from PIL import Image
import torch

from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import CLIPModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model.save_pretrained("./clip-model")

model = CLIPModel.from_pretrained("./clip-model").to(device).eval()

# model_path = "/media/external_10TB/10TB/p_haghighi/clip-model"
# model = CLIPModel.from_pretrained(model_path, from_tf=False).to(device).eval()

print(f'CLIP model loaded on {device}.')

def process_images(images):
    transform = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    preprocessed_images = [transform(img) for img in images]
    return preprocessed_images

sampled_frames_path = '/media/10TB71/ramezani/ACTIVITY_NET/sampled_frames/'
embeddings_path = '/media/10TB71/ramezani/CLIP_embeddings/'


dataset_paths = [
    "v1-3/test",
    "v1-3/train_val",
    "v1-2/train",
    "v1-2/test",
    "v1-2/val",
]

for dataset_path in dataset_paths:
    print(dataset_path)
    # os.makedirs(embeddings_path+dataset_path, exist_ok=True)
    
    number_of_videos = len(os.listdir(sampled_frames_path+dataset_path))
    
    print(f"Calculate CLIP embedding of {dataset_path} ({number_of_videos} videos) started!")

    for i, folder_name in enumerate(os.listdir(sampled_frames_path+dataset_path)):
        folder_path = os.path.join(sampled_frames_path+dataset_path, folder_name)
        embeddings_path_file = os.path.join(embeddings_path+dataset_path, f'{folder_name}.npy')
        
        if not os.path.isdir(folder_path):
            # print(f'{folder_name} is not exist!')
            continue

        if os.path.exists(embeddings_path_file):
            # print(f'{folder_name}".npy" has already calculated!')
            continue

        images = []

        for image_name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                image = Image.open(image_path)
                images.append(image)
        
        tic = time.time()
        preprocessed_images = process_images(images)
        toc = time.time()

        torch.cuda.empty_cache() 

        image_inputs = torch.squeeze(torch.tensor(np.stack(preprocessed_images)), dim=1).to(device)
        
        tuc = time.time()

        with torch.no_grad():
            embeddings = model.get_image_features(image_inputs).float()
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        
        tyc = time.time()

        np.save(embeddings_path_file, embeddings.cpu().numpy())

        print(f'[{i}/{number_of_videos}]: {len(images):3d} images | Preprocessing: {toc-tic:2.2f}s | Squeezing: {tuc-toc:2.2f} | CLIP: {tyc-tuc:2.2f}s | Saving: {time.time() - tyc:.2f}s')

    print(f"Calculate CLIP embedding of {dataset_path} ({number_of_videos} videos) ended!")