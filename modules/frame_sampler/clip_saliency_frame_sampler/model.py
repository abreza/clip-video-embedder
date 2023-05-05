import timm
import torch
import numpy as np
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, preprocessed_images):
        features = self.model(preprocessed_images)
        x = features.view(features.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class CLIPTeacher(nn.Module):
    def __init__(self):
        super(CLIPTeacher, self).__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14')

    def forward(self, preprocessed_images, descriptions):
        image_inputs = torch.squeeze(torch.tensor(np.stack(preprocessed_images)), dim=1)

        text_inputs = self.processor(text=descriptions, return_tensors="pt", padding=True)
        
        image_features = self.model.get_image_features(image_inputs).float()
        text_features = self.model.get_text_features(**text_inputs).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        max_similarity = np.max(similarities,axis=0)
        
        return max_similarity

