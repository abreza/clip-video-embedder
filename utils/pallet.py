import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_text_image_pallet(model, processor, images, sentences):

    processed_images = processor(images =images)['pixel_values']

    image_input = torch.squeeze(torch.tensor(np.stack(processed_images)), dim=1).to(model.device)
    text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        image_features = model.get_image_features(image_input).float()
        text_features = model.get_text_features(**text_inputs).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    plt.figure(figsize=(10, 10))
    plt.imshow(similarity, vmin=np.min(similarity), vmax=np.max(similarity))
    plt.yticks(range(similarity.shape[0]), sentences, fontsize=18)
    plt.xticks([])

    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=10)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, similarity.shape[1] - 0.5]);
    plt.ylim([similarity.shape[0] - 0.5, -1.6]);


def plot_image_image_pallet(model, processor, images):  
   
    processed_images = processor(images=images)['pixel_values']

    image_input = torch.squeeze(torch.tensor(np.stack(processed_images)), dim=1).to(model.device)

    with torch.no_grad():
        image_features = model.get_image_features(image_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = image_features.cpu().numpy() @ image_features.cpu().numpy().T

    plt.figure(figsize=(10, 10))
    plt.imshow(similarity, vmin=np.min(similarity), vmax=np.max(similarity))

    for i, image in enumerate(images):
        plt.imshow(image, extent=(-1.6, -0.6, i - 0.5, i + 0.5), origin="lower")
        
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=10)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.yticks([])
    plt.xticks([])

    plt.xlim([-1.6, similarity.shape[1] - 0.5]);
    plt.ylim([similarity.shape[0] - 0.5, -1.6]);