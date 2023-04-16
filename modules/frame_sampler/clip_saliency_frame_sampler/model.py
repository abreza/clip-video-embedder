import timm
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        self.fc1 = nn.Linear(self.model.num_features, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model.forward_features(images)
        x = self.fc1(features)
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

    def forward(self, images, descriptions):
        inputs = self.processor(
            text=descriptions, images=images, return_tensors="pt", padding=True
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image

        average_scores = logits_per_image.mean(dim=1)

        return average_scores
