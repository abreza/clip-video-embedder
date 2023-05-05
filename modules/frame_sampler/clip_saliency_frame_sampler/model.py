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
        features = self.model.forward_features(preprocessed_images)
        x = features.view(features.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return torch.flatten(x)


class CLIPTeacher(nn.Module):
    def __init__(self, device):
        super(CLIPTeacher, self).__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14')
        self.device = device

    def forward(self, preprocessed_images, descriptions):

        text_inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(self.device)
        
        image_features = self.model.get_image_features(preprocessed_images).float()
        text_features = self.model.get_text_features(**text_inputs).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = torch.matmul(text_features, image_features.T)

        max_similarity = torch.max(similarities, axis=0)

        return max_similarity

