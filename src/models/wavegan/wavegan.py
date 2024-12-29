"""
This code is an adaptation of the original WaveGAN model from the repository:
https://github.com/chrisdonahue/wavegan.

The original WaveGAN was designed to generate and discriminate 1D waveform data, typically used for audio generation. This adaptation modifies the original architecture to make it suitable for time-series data, allowing it to handle different types of sequential data beyond audio signals, such as financial, biomedical, or sensor data.

Key Modifications:
1. **Transpose1dLayer**: This class handles both transposed convolutions (for upsampling) and standard 1D convolutions. It was adjusted to handle the possibility of upsampling the time-series data. The reflection padding and upsampling logic have been adapted to better fit time-series data structures.

2. **WaveGANGenerator**: The generator network creates synthetic time-series data from a latent space. Instead of generating waveform-like audio, this network generates time-series sequences. 
   - The `latent_dim` controls the dimensionality of the input noise vector, and the fully connected layer (`fc1`) maps the latent space to a higher-dimensional feature space.
   - The successive `deconv_` (transposed convolution) layers are designed to progressively expand the feature maps, creating larger time-series sequences. The layers also include an option for upsampling, which allows the model to generate longer sequences.

3. **WaveGANDiscriminator**: This component is designed to distinguish between real and generated time-series data. It uses convolutional layers (`conv1` to `conv5`) to process the time-series data and extract hierarchical features.
   - The `PhaseShuffle` modules are used to add invariance to phase shifts, a technique originally designed for audio data but repurposed here for time-series analysis.
   - The final fully connected layer (`fc1`) outputs a probability score indicating whether the input time-series is real or fake.

4. **PhaseShuffle and PhaseRemove**: The phase shuffle technique introduces random shifts in the feature axis of the time-series data, enhancing the model's ability to generalize across different time shifts. This operation is a key part of the discriminator, allowing it to be more robust to small variations in input data.
   
Overall, this implementation preserves the original structure and functionality of WaveGAN but adapts it to time-series generation and classification. The modifications include:
- Adjusting the kernel sizes, strides, and padding values to better suit the nature of time-series data.
- Reworking the generator and discriminator to be more applicable to sequential data rather than audio-specific features.
- Incorporating methods such as phase shuffling that have been demonstrated to be effective in training models on 1D data like audio but also generalizable to other time-series tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# Adapted 1D Transpose Layer for Time Series
class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample) if upsample else None
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class TimeSeriesGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, latent_dim=100, post_proc_filt_len=512, verbose=False, upsample=True):
        super(TimeSeriesGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.latent_dim = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        self.fc1 = nn.Linear(latent_dim, 256 * model_size)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4
        self.deconv_1 = Transpose1dLayer(16 * model_size, 8 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(8 * model_size, 4 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(4 * model_size, 2 * model_size, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer(2 * model_size, model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Transpose1dLayer(model_size, num_channels, 25, stride, upsample=upsample)

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_1(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_2(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_3(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_4(x))
        if self.verbose:
            print(x.shape)

        output = F.tanh(self.deconv_5(x))
        return output

# Phase Shuffle for Time Series
class PhaseShuffle(nn.Module):
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        x_shuffle = x.clone()
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle

# Time Series Discriminator
class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2, alpha=0.2, verbose=False):
        super(TimeSeriesDiscriminator, self).__init__()
        self.model_size = model_size
        self.ngpus = ngpus
        self.num_channels = num_channels
        self.shift_factor = shift_factor
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)
    
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2, alpha=0.2, verbose=False):
        super(TimeSeriesDiscriminator, self).__init__()
        self.model_size = model_size
        self.ngpus = ngpus
        self.num_channels = num_channels
        self.shift_factor = shift_factor
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)