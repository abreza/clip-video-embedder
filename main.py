from utils.dotdict import DotDict

from dataloaders.data_dataloaders import DATALOADER_DICT

from transformers import CLIPTokenizer

from configs.MSRVTT.loader_train_9k import params as MSRVTT_DATA_PARAMS


def main():
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    args = dict(max_words=32,
                feature_framerate=1,
                max_frames=100,
                expand_msrvtt_sentences=True,
                train_frame_order=0,
                slice_framepos=2,
                batch_size=128,
                n_gpu=1,
                num_thread_reader=0,)

    train_args = DotDict(dict(MSRVTT_DATA_PARAMS, **args))

    train_dataloader, train_length = DATALOADER_DICT['msrvtt']["train"](
        train_args, tokenizer)


if __name__ == '__main__':
    main()
