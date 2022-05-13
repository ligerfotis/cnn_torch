verbose = False
resume = False
batch_size = 1024
epochs = 10
lr = 1e-5
weight_decay = 1e-8
kernel_size = 3
num_workers = 16
save_integral = 10
beta = 1
latent_space_dimension = 16
# input_height = 28
# input_width = 28
# dataset = "MNIST"
dataset = "CIFAR10"
# dense_encoder_hidden_layers = [128, 64, 36, 18, 9]
# dense_decoder_hidden_layers = [9, 18, 36, 64, 128]
dense_encoder_hidden_layers = [256, 128, 64, 36, 18]
dense_decoder_hidden_layers = [18, 36, 64, 128, 256]
conv_encoder_hidden_layers = [16, 32, 64, 128, 256]
conv_decoder_hidden_layers = [128, 64, 32, 16, 8]
# conv_encoder_hidden_layers = [128, 256, 512]
# conv_decoder_hidden_layers = [256, 128, 64]
# encoder_hidden_layers = [128, 64, 36, 18, 9]
# decoder_hidden_layers = [9, 18, 36, 64, 128]

if dataset in ["MNIST", "mnist", "Mnist"]:
    input_height = 28
    input_width = 28
    in_channel_size = 1

elif dataset in ["CIFAR10", "cifar10", "Cifar10", "CIFAR-10"]:
    input_height = 32
    input_width = 32
    in_channel_size = 3