from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_pallet(model, processor, images=None, sentences=None):
    
    model.to(device).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if images == None:
        images = []
        processed_images = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        # color_names = ['red','green','blue','yellow','light blue']

        for color in colors:
            image = Image.new('RGB', (224, 224), color)
            images.append(image)
            processed_images.append(processor(images =image)['pixel_values'])

    image_input = torch.squeeze(torch.tensor(np.stack(processed_images)), dim=1).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(image_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)


    if sentences:

        text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs).float()

        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        plt.yticks(range(similarity.shape[0]), sentences, fontsize=18)
        plt.xticks([])

        for i, image in enumerate(images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.3f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, similarity.shape[1] - 0.5]);
        plt.ylim([similarity.shape[0] - 0.5, -1.6]);


    else:
        similarity = image_features.cpu().numpy() @ image_features.cpu().numpy().T


    plt.figure(figsize=(14, 10))
    plt.imshow(similarity, vmin=0.9, vmax=1)

    for i, image in enumerate(images):
        plt.imshow(image, extent=(-1.6, -0.6, i - 0.5, i + 0.5), origin="lower")
        
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")


    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.3f}", ha="center", va="center", size=12)


    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.yticks([])
    plt.xticks([])

    plt.xlim([-1.6, similarity.shape[1] - 0.5]);
    plt.ylim([similarity.shape[0] - 0.5, -1.6]);