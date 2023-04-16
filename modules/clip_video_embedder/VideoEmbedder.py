import numpy as np
from transformers import CLIPProcessor, CLIPModel


class VideoEmbedder:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def uniform_sampling(self, frames, num_samples):
        frame_count = len(frames)
        indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        return sampled_frames

    def get_frame_embeddings(self, frames):
        inputs = self.processor(images=frames, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        frame_embeddings = outputs.detach().numpy()
        return frame_embeddings

    def aggregate_embeddings(self, embeddings, strategy="mean"):
        if strategy == "mean":
            return np.mean(embeddings, axis=0)
        elif strategy == "average":
            return np.average(embeddings, axis=0)
        else:
            raise ValueError("Invalid aggregation strategy")

    def embed_video(self, frames, num_samples, aggregation_strategy="mean"):
        sampled_frames = self.uniform_sampling(frames, num_samples)
        frame_embeddings = self.get_frame_embeddings(sampled_frames)
        aggregated_embedding = self.aggregate_embeddings(
            frame_embeddings, aggregation_strategy)
        return aggregated_embedding

    def get_text_embedding(self, text):
        text_inputs = self.processor(
            text=text, return_tensors="pt", padding=True)
        text_outputs = self.model.get_text_features(**text_inputs)
        text_embedding = text_outputs.detach().numpy()
        return text_embedding

    def similarity(self, video_embedding, text_embedding):
        similarity_score = np.dot(video_embedding, text_embedding.T)
        return similarity_score
