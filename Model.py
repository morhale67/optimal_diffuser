import pylops
import torch
from torch import nn
import math
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResConv(nn.Module):
    def __init__(self, img_dim, n_masks):
        super(MyModel, self).__init__()
        pic_width = int(np.sqrt(img_dim))

        self.conv1 = ConvBlock(1, 32)
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 256)

        fc_input_size = 256 * (pic_width // 16) * (pic_width // 16)

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc_diff = nn.Linear(256, n_masks * img_dim)
        self.prob_vector1 = nn.Linear(128, 10)
        self.prob_vector2 = nn.Linear(128, 10)
        self.prob_vector3 = nn.Linear(128, 10)
        self.prob_vector4 = nn.Linear(128, 10)

    def forward(self, x_i):
        x = self.conv1(x_i)
        x = self.res1(x) + x
        x = self.res2(x) + x
        x = self.res3(x) + x

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)

        diffuser = self.fc_diff(x)
        diffuser = torch.sigmoid(diffuser)

        prob_vector1 = self.prob_vector1(x)
        prob_vector2 = self.prob_vector2(x)
        prob_vector3 = self.prob_vector3(x)
        prob_vector4 = self.prob_vector4(x)
        prob_vectors = [prob_vector1, prob_vector2, prob_vector3, prob_vector4]

        return diffuser, prob_vectors




class MyModel(nn.Module):
    def __init__(self, img_dim, n_masks):
        super(MyModel, self).__init__()
        pic_width = int(np.sqrt(img_dim))
        # Image processing layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Adjusting input size for fully connected layers
        fc_input_size = 64 * (pic_width // 4) * (pic_width // 4)  # Adjust input size based on the image dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output layers for diffuser and probabilistic vectors
        self.fc_diff = nn.Linear(128, n_masks*img_dim)
        self.prob_vector1 = nn.Linear(128, 10)  # Adjust output size based on the number of classes
        self.prob_vector2 = nn.Linear(128, 10)
        self.prob_vector3 = nn.Linear(128, 10)
        self.prob_vector4 = nn.Linear(128, 10)

    def forward(self, x_i):
        x = self.conv1(x_i)
        # if torch.isnan(x).any().item():
        #     print(x)

        x = self.bn1(x)
        x = torch.relu(x)

        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)


        diffuser = self.fc_diff(x)
        diffuser = torch.sigmoid(diffuser)  # Apply sigmoid activation for diffuser

        prob_vector1 = self.prob_vector1(x)
        prob_vector2 = self.prob_vector2(x)
        prob_vector3 = self.prob_vector3(x)
        prob_vector4 = self.prob_vector4(x)
        prob_vectors = [prob_vector1, prob_vector2, prob_vector3, prob_vector4]

        return diffuser, prob_vectors


class Gen_with_p_sb(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, n_masks * img_dim)
        self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear2_params = nn.Linear(256, 64)
        self.relu2_params = nn.ReLU()

        self.linear_params = nn.Linear(64, 4)  # 4 parameters: maxiter, niter_inner, alpha, total_variation_rate
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)


        diffuser = self.linear3(x)
        diffuser = self.bn3(diffuser)
        diffuser = self.sigmoid(diffuser)

        # Compute path for parameters
        params_x = self.linear2_params(x)
        params_x = self.relu2_params(params_x)
        params_x = self.linear_params(params_x)
        params_x = self.softmax(params_x)

        return


class Gen(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, n_masks * img_dim)
        self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.sigmoid(x)
        return x


class Masks4(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.d_output = n_masks * img_dim
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, self.d_output),
            nn.BatchNorm1d(self.d_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Diff4(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.d_output = n_masks * img_dim
        self.model = nn.Sequential(
            nn.Linear(z_dim, self.d_output // 8),
            nn.BatchNorm1d(self.d_output // 8),
            nn.ReLU(),

            nn.Linear(self.d_output // 8, self.d_output // 4),
            nn.BatchNorm1d(self.d_output // 4),
            nn.ReLU(),

            nn.Linear(self.d_output // 4, self.d_output // 2),
            nn.BatchNorm1d(self.d_output // 2),
            nn.ReLU(),

            nn.Linear(self.d_output // 2, self.d_output),
            nn.BatchNorm1d(self.d_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



# Define the custom layer
class ElementwiseMultiplyLayer(nn.Module):
    def __init__(self, input_size, n_mask):
        super(ElementwiseMultiplyLayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(n_mask, input_size))  # Define n_mask sets of learnable weights

    def forward(self, input_vector):
        # Perform element-wise multiplication for each set of weights
        output_vectors = self.weights * input_vector.unsqueeze(0)  # Add a batch dimension
        buckets = torch.sum(output_vectors, dim=1)  # Calculate the sum along dim=1
        return buckets

class Gen_no_batch(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, n_masks * img_dim// 2)
        self.bn3 = nn.BatchNorm1d(n_masks * img_dim // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        # x = torch.sign(x)
        out = self.sigmoid(x)

        return out


class Gen_big_diff(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks, ac_stride):
        super().__init__()
        self.ac_stride = ac_stride
        self.pic_width = int(math.sqrt(img_dim))
        self.linear1 = nn.Linear(z_dim, img_dim)
        self.bn1 = nn.BatchNorm1d(img_dim)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(img_dim, img_dim)
        self.bn2 = nn.BatchNorm1d(img_dim)
        self.relu2 = nn.ReLU()

        self.diff_width = self.pic_width + self.ac_stride * n_masks
        self.linear3 = nn.Linear(img_dim, self.diff_width*self.pic_width)
        self.bn3 = nn.BatchNorm1d(self.diff_width*self.pic_width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.sigmoid(x)
        return x


class Gen_conv_3(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.img_dim = img_dim
        self.n_masks = n_masks
        self.pic_width = int(math.sqrt(self.img_dim))
        self.linear1 = nn.Linear(z_dim, self.img_dim)
        self.bn1 = nn.BatchNorm1d(self.img_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(1, self.n_masks // 4, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(self.n_masks//4)

        self.conv3 = nn.Conv2d(self.n_masks // 4, self.n_masks//2, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(self.n_masks//2)

        self.conv4 = nn.Conv2d(self.n_masks//2, self.n_masks, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(self.n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 1, self.pic_width, self.pic_width)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sigmoid(x)
        return x


class Gen_conv1(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.img_dim = img_dim
        self.n_masks = n_masks
        self.pic_width = int(math.sqrt(self.img_dim))
        self.linear1 = nn.Linear(z_dim, self.img_dim)
        self.bn1 = nn.BatchNorm1d(self.img_dim)
        self.relu1 = nn.ReLU()
        #
        # self.linear2 = nn.Linear(128, img_dim)
        # self.bn2 = nn.BatchNorm1d(img_dim)
        # self.relu2 = nn.ReLU()

        # self.linear3 = nn.Linear(img_dim, n_masks * img_dim)
        # self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
        # self.sigmoid = nn.Sigmoid()

        # Change the last linear layer to a convolutional layer
        self.conv3 = nn.Conv2d(1, self.n_masks, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(self.n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 1, self.pic_width, self.pic_width)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x.reshape(-1, self.n_masks, self.img_dim)
        x = self.sigmoid(x)
        return x


class Diff_Paths(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.fc_layers = nn.ModuleList([
            nn.Linear(256, img_dim) for _ in range(n_masks)
        ])
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)


        # Forward through each fully connected layer in the ModuleList
        paths = [fc_layer(x) for fc_layer in self.fc_layers]
        # Concatenate the outputs along the last dimension (dim=1)
        x = torch.cat(paths, dim=1)
        x = self.sigmoid(x)
        return x

