
from dataloaders.data_dataloaders import dataloader_activity_train


def main():
    
    args = dict(data_path='',
                features_path='',
                max_words=77,
                feature_framerate=1,
                max_frames=100,
                train_frame_order=,
                slice_framepos=2,
                batch_size=128,
                n_gpu=1,
                num_thread_reader=0)
    

    train_dataloader, train_length = dataloader_activity_train(args)
    print(train_dataloader)



if __name__ == '__main__':
    main()
