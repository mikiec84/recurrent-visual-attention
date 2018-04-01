import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


class retina(object):
    def __init__(self, patch_size, num_patches, scale):
        """
        @param patch_size: side length of the extracted patched.
        @param num_patches: number of patches to extract in the glimpse.
        @param scale: scaling factor that controls the size of successive patches.
        """
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.scale = scale

    def foveate(self, x, l):
        """
        Extract `num_patches` square patches,  centered at location `l`.
        The initial patch is a square of sidelength `patch_size`,
        and each subsequent patch is a square whose sidelength is `scale`
        times the size of the previous patch.  All patches are finally
        resized to the same size of the first patch and then flattened.

        @param x: img. (batch, height, width, channel)
        @param l: location. (batch,2)
        @return Variable: (batch, num_patches*channel*patch_size*patch_size).
        """
        patches = []
        size = self.patch_size

        # extract num_patches patches of increasing size
        for i in range(self.num_patches):
            patches.append(self.extract_patch(x, l, size))
            size = int(self.scale * size)

        # resize the patches to squares of size patch_size
        for i in range(1, len(patches)):
            num_patches = patches[i].shape[-1] // self.patch_size
            patches[i] = F.avg_pool2d(patches[i], num_patches)

        # concatenate into a single tensor and flatten
        patches = torch.cat(patches, 1)
        patches = patches.view(patches.shape[0], -1)

        return patches

    def extract_patch(self, x, l, size):
        """
        @param x: img. (batch, channel, height, width)
        @param l: location. (batch, 2)
        @param size: the size of the extracted patch.
        @return Variable (batch, channel, size, size)
        """
        B, C, H, W = x.shape

        if not hasattr(self, 'imgShape'):
            self.imgShape = torch.FloatTensor([H, W]).unsqueeze(0)

        # coordins from [-1,1] to H,W scale
        coords = (0.5 * ((l.data + 1.0) * self.imgShape)).long()

        # pad the image with enough 0s
        x = nn.ConstantPad2d(size//2, 0.)(x)

        # calculate coordinate for each batch samle (padding considered)
        from_x, from_y = coords[:, 0], coords[:, 1]
        to_x, to_y = from_x + size, from_y + size

        # extract the patches
        patch = []
        for i in range(B):
            patch.append(x[i, :, from_y[i]:to_y[i], from_x[i]:to_x[i]].unsqueeze(0))

        return torch.cat(patch)


class glimpse_network(nn.Module):
    def __init__(self, hidden_g, hidden_l, patch_size, num_patches, scale, num_channel):
        """
        @param hidden_g: hidden layer size of the fc layer for `phi`.
        @param hidden_l: hidden layer size of the fc layer for `l`.
        @param patch_size: size of the square patches in the glimpses extracted
        @param by the retina.
        @param num_patches: number of patches to extract per glimpse.
        @param scale: scaling factor that controls the size of successive patches.
        @param num_channel: number of channels in each image.
        """
        super(glimpse_network, self).__init__()
        self.retina = retina(patch_size, num_patches, scale)

        # glimpse layer
        D_in = num_patches*patch_size*patch_size*num_channel
        self.fc1 = nn.Linear(D_in, hidden_g)

        # location layer
        self.fc2 = nn.Linear(2, hidden_l)

        self.fc3 = nn.Linear(hidden_g, hidden_g+hidden_l)
        self.fc4 = nn.Linear(hidden_l, hidden_g+hidden_l)

    def forward(self, x_t, l_t):
        """
        @param x_t: (batch, height, width, channel)
        @param l_t: (batch, 2)
        @return output: (batch, hidden_g+hidden_l)
        """
        glimpse = self.retina.foveate(x_t, l_t)

        what = self.fc3(F.relu(self.fc1(glimpse)))
        where = self.fc4(F.relu(self.fc2(l_t)))

        g = F.relu(what + where)

        return g


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean
        mu = F.tanh(self.fc(h_t))

        # sample from gaussian parametrized by this mean
        noise = torch.from_numpy(np.random.normal(
            scale=self.std, size=mu.shape)
        )
        noise = Variable(noise.float()).type_as(mu)
        l_t = mu + noise

        # bound between [-1, 1]
        l_t = F.tanh(l_t)

        # prevent gradient flow
        l_t = l_t.detach()

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t))
        return b_t
