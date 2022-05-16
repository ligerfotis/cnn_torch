### Run the Experiment

    python train.py -m small -bs 512 -e 10 -lr 1e-5 -nw 16
or

    python3 train.py -m big -bs 4096 -e 100 -lr 1e-5 -nw 16

### Using docker

    docker run -it --gpus all --shm-size="16g" --name train_cnn ligerfotis/torch1.8.2_cu11.1_opencv4.5:latest