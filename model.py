import torch
from model_utils import create_conv_encoder

kernel = 3
padding = 1
stride = 1


class CNN_classifier(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers = []
        # self.hidden_layers = [64, 128, 256, 512, 512]
        # hidden_layers = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]        # VGG11
        self.hidden_layers = [64, 'M', 128, 'M', 256, 256]
        # hidden_layers = [64, 128, 256, 512]
        in_channels = 3
        adaptive_pooling_features = 4
        linear_dim = adaptive_pooling_features * adaptive_pooling_features * self.hidden_layers[-1]
        self.encoder_cnn, self.feature_map_size = create_conv_encoder(in_channels=in_channels,
                                                                      hidden_layers=self.hidden_layers,
                                                                      device=device)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((adaptive_pooling_features, adaptive_pooling_features))
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(linear_dim, linear_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(linear_dim, linear_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(linear_dim, 10),
        )

    def forward(self, x):
        conv_x = self.encoder_cnn(x)
        adaptive_mpooling = self.avgpool(conv_x)
        flatten_x = self.flatten(adaptive_mpooling)
        logits = self.classifier(flatten_x)
        return logits
