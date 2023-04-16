from datasets.msrvtt.dataset import MSRVTTDataset
from modules.clip_video_embedder import VideoEmbedder


def main():
    dataset = MSRVTTDataset()
    video_embedder = VideoEmbedder()

    for idx in range(len(dataset)):
        frames, captions = dataset[idx]
        video_embedding = video_embedder.embed_video(frames, num_samples=10)

        for caption in captions:
            text_embedding = video_embedder.get_text_embedding(caption)
            similarity_score = video_embedder.similarity(
                video_embedding, text_embedding)
            print(
                f"Similarity between video and caption '{caption}': {similarity_score}")


if __name__ == "__main__":
    main()
