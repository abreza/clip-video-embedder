from utils.dotdict import DotNotationDict

from dataloaders.data_dataloaders import DATALOADER_DICT

from transformers import CLIPTokenizer


def main():
    tokenizer = CLIPTokenizer.from_pretrained("clip-vit-large-patch14")

    train_args = DotNotationDict({
        "train_csv": 'MSRVTT_train.9k.csv',
        "data_path": 'MSRVTT_data.json',
        "features_path": 'MSRVTT_Videos',
        "max_words": 32,
        "feature_framerate": 1,
        "max_frames": 100,
        "expand_msrvtt_sentences": True,
        "train_frame_order": 0,
        "slice_framepos": 2,
        "batch_size": 128,
        "n_gpu": 1,
        "num_thread_reader": 0,
    })

    train_dataloader, train_length = DATALOADER_DICT['msrvtt']["train"](
        train_args, tokenizer)


if __name__ == '__main__':
    main()
