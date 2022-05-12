import torch
from model_utils import create_conv_encoder

kernel = 3
padding = 1
stride = 1


class CNN_classifier(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers = []
        self.hidden_layers = [8, 16, 64, 128, 256, 512]
        in_channels = 3
        adaptive_pooling_features = 7
        self.encoder_cnn, self.feature_map_size = create_conv_encoder(in_channels=in_channels,
                                                                      hidden_layers=self.hidden_layers,
                                                                      device=device)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((adaptive_pooling_features, adaptive_pooling_features))
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(adaptive_pooling_features * adaptive_pooling_features * self.hidden_layers[-1], 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        conv_x = self.encoder_cnn(x)
        adaptive_mpooling = self.avgpool(conv_x)
        flatten_x = self.flatten(adaptive_mpooling)
        logits = self.classifier(flatten_x)
        return logits
