import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicNet(nn.Module):
    def __init__(self, h, dropout_p):
        # print(h_in, h_out, dropout_p)
        super(BasicNet, self).__init__()

        self.conv_seq = nn.Sequential(
                        nn.Conv2d(3, 64, 3),
                        nn. ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(256, 512, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.dense_seq = nn.Sequential(
                         nn.Linear(51201, h),
                         nn.ReLU(),
                         nn.Dropout(p=dropout_p),
                         nn.Linear(h, h),
                         nn.ReLU(),
                         nn.Dropout(p=dropout_p),
                         nn.Linear(h, 100),
                         nn.ReLU()
        )

    def forward(self, x, ecm):

        x = self.conv_seq(x)

        x = x.view(x.shape[0], -1)
        x = torch.hstack((x, ecm.view(-1, 1).float()))
        # print(x.shape)

        x = self.dense_seq(x)

        return x


class Resnet18(nn.Module):
    def __init__(self, output=100, pretrained=False, relu=False):
        super(Resnet18, self).__init__()

        self.res = models.resnet18(pretrained=pretrained)
        num_features = self.res.fc.in_features
        self.res.fc = nn.Linear(num_features, output)

        self.relu = relu
        self.r = nn.ReLU()

    def forward(self, x, ecm):
        x = self.res(x)
        if self.relu:
            x = self.r(x)

        return x


class Resnet18Hidden(nn.Module):
    def __init__(self, output=100, hidden=256, pretrained=False, relu=False):
        super(Resnet18Hidden, self).__init__()

        self.res = models.resnet18(pretrained=pretrained)
        num_features = self.res.fc.in_features
        self.res.fc = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU(), nn.Linear(hidden, output))

        self.relu = relu
        self.r = nn.ReLU()

    def forward(self, x, ecm):
        x = self.res(x)
        if self.relu:
            x = self.r(x)

        return x


class Resnet18HiddenDropout(nn.Module):
    def __init__(self, output=100, hidden=256, p=0.5, pretrained=False, relu=False, ecm=False):
        super(Resnet18HiddenDropout, self).__init__()
        self.ecm = ecm

        self.res = models.resnet18(pretrained=pretrained)
        num_features = self.res.fc.out_features
        if ecm:
            self.fc = nn.Sequential(nn.ReLU(), nn.Linear(num_features + 1, hidden), nn.ReLU(True), nn.Dropout(p),
                                        nn.Linear(hidden, output))
        else:
            self.fc = nn.Sequential(nn.ReLU(True), nn.Linear(num_features, hidden), nn.ReLU(True), nn.Dropout(p), nn.Linear(hidden, output))

        # del self.res.fc
        # self.res.fc = None

        self.relu = relu
        self.r = nn.ReLU()

    def forward(self, x, ecm):
        x = self.res(x)

        # print(x.shape)
        # print(ecm.shape)
        if self.ecm:
            x = torch.hstack([x, ecm.unsqueeze(-1).float()])
            # print(x.shape)
        x = self.fc(x)

        if self.relu:
            x = self.r(x)

        return x


class VGG11BN(nn.Module):
    def __init__(self, output=100, pretrained=False):
        super(VGG11BN, self).__init__()

        self.vgg = models.vgg11_bn(pretrained=pretrained)
        num_features = self.vgg.classifier[6].in_features
        features = list(self.vgg.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, output)])  # Add our layer with 2 outputs
        self.vgg.classifier = nn.Sequential(*features)

    def forward(self, x, ecm):
        return self.vgg(x)


class Encoder(nn.Module):

    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()

        ## Convolutional section
        # self.encoder_cnn = nn.Sequential(
        #     nn.Conv2d(3, 32, 4, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, 4, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 512, 4, stride=2, padding=0),
        #     nn.ReLU(True),
        # )

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(256, 512, 4, stride=2, padding=0),
            # nn.ReLU(True),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(12 * 12 * 128, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 12 * 12 * 128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(128, 12, 12))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1, output_padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class BasicAutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim, fc2_input_dim)
        # self.decoder = Decoder(encoded_space_dim, fc2_input_dim)
        self.decoder = HalfResizeDecoder(encoded_space_dim, fc2_input_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x


class ResizeConvEncoder(nn.Module):

    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn. ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(256, 512, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
        )

        ## Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 512, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        # print(x)
        x = self.encoder_cnn(x)
        # print(x.shape)
        x = self.flatten(x)
        # # print(x.shape)
        x = self.encoder_lin(x)

        return x


class ResizeConvDecoder(nn.Module):

    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 4 * 4 * 512),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(512, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2), #, mode='bilinear'),
            nn.ZeroPad2d(2),
            nn.Conv2d(512, 256, 3, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=1),
            # nn.Upsample(scale_factor=2), #, mode='bilinear'),
            # nn.ZeroPad2d(3),
            # nn.Conv2d(256, 128, 3, padding=0),
            # # # # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            # nn.Upsample(scale_factor=2), #, mode='bilinear'),
            # nn.ZeroPad2d(2),
            # nn.Conv2d(128, 64, 3, padding=0),
            # # # # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Upsample(scale_factor=2), #, mode='bilinear'),
            # nn.ZeroPad2d(1),
            # nn.Conv2d(64, 32, 3, padding=0),
            # # # # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.Upsample(scale_factor=2),  #mode='bilinear'),
            # nn.ZeroPad2d(1),
            # nn.Conv2d(32, 3, 3, padding=0),
            # # # nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        # # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.decoder_conv(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        # print(x)
        return x


class ResizeConvAutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder = ResizeConvEncoder(encoded_space_dim, fc2_input_dim)
        self.decoder = ResizeConvDecoder(encoded_space_dim, fc2_input_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x


class HalfResizeDecoder(nn.Module):

    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 12 * 12 * 128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(128, 12, 12))

        self.decoder_conv = nn.Sequential(
            # nn.Upsample(scale_factor=2),  # , mode='bilinear'),
            # nn.ZeroPad2d(2),
            # nn.Conv2d(512, 256, 3, padding=0),
            # nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),  # mode='bilinear'),
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # mode='bilinear'),
            # nn.ZeroPad2d(1),
            nn.Conv2d(64, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0),
            nn.Upsample(scale_factor=2),  # mode='bilinear'),
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=0),
            nn.Upsample(scale_factor=2),  #mode='bilinear'),
            nn.ZeroPad2d(1),
            nn.Conv2d(16, 3, 3, padding=0),
            # nn.ReLU(True),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x