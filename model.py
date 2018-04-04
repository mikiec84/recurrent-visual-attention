from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import BaselineNet
from modules import GlimpseNet, core_network
from modules import ActionNet, LocationNet


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(self, args):
        """
        Initialize the recurrent attention model and its
        different components.
        """
        super(RecurrentAttention, self).__init__()
        self.std = args.std
        self.hidden_size = args.hidden_size
        self.use_gpu = args.use_gpu
        self.num_glimpses = args.num_glimpses
        self.M = args.M

        self.sensor = GlimpseNet(args.glimpse_hidden, args.loc_hidden, args.patch_size, args.num_patches, args.glimpse_scale, args.num_channels)
        self.rnn = core_network(args.hidden_size, args.hidden_size)
        self.locator = LocationNet(args.hidden_size, 2, args.std)
        self.classifier = ActionNet(args.hidden_size, args.num_classes)
        self.baseliner = BaselineNet(args.hidden_size, 1)
        self.name = 'ram_{}_{}x{}_{}'.format(args.num_glimpses, args.patch_size, args.patch_size, args.glimpse_scale)

    def step(self, x, l_t_prev, h_t_prev, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a 2D vector of shape (B, 1). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        mu, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, None, log_pi

    def forward(self, x, y, is_training=True):
        if self.use_gpu:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        if not is_training:
            return self.forward_test(x, y)

        # initialize location vector and hidden state
        h_t, l_t = self.rnn.init_state(x.shape[0], self.use_gpu)

        # extract the glimpses
        locs = []
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses):
            # Note that log_probas is None except for t=num_glimpses-1
            h_t, l_t, b_t, log_probas, p = self.step(x, l_t, h_t, last=(t==self.num_glimpses-1))

            # store
            locs.append(l_t)
            baselines.append(b_t)
            log_pi.append(p)

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        # calculate reward
        predicted = torch.max(log_probas, 1)[1]
        R = (predicted.detach() == y).float()
        R = R.unsqueeze(1).repeat(1, self.num_glimpses)

        # compute losses for differentiable modules
        loss_action = F.nll_loss(log_probas, y)
        loss_baseline = F.mse_loss(baselines, R)

        # compute reinforce loss
        adjusted_reward = R - baselines.detach()
        loss_reinforce = torch.mean(-log_pi*adjusted_reward)

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce

        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return {'loss': loss,
                'acc': acc,
                'locs': locs,
                'x': x}

    def forward_test(self, x, y):
        # duplicate 10 times
        x = x.repeat(self.M, 1, 1, 1)

        # initialize location vector and hidden state
        h_t, l_t = self.rnn.init_state(x.shape[0])

        # extract the glimpses
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses):

            # forward pass through model
            h_t, l_t, b_t, log_probas, p = self.step(x, l_t, h_t, last=(t==self.num_glimpses-1))

            # store
            baselines.append(b_t)
            log_pi.append(p)

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        # average
        log_probas = log_probas.view(
            self.M, -1, log_probas.shape[-1]
        )
        log_probas = torch.mean(log_probas, dim=0)

        baselines = baselines.contiguous().view(
            self.M, -1, baselines.shape[-1]
        )
        baselines = torch.mean(baselines, dim=0)

        log_pi = log_pi.contiguous().view(
            self.M, -1, log_pi.shape[-1]
        )
        log_pi = torch.mean(log_pi, dim=0)

        # calculate reward
        predicted = torch.max(log_probas, 1)[1]
        R = (predicted.detach() == y).float()
        R = R.unsqueeze(1).repeat(1, self.num_glimpses)

        # compute losses for differentiable modules
        loss_action = F.nll_loss(log_probas, y)
        loss_baseline = F.mse_loss(baselines, R)

        # compute reinforce loss
        adjusted_reward = R - baselines.detach()
        # https://github.com/kevinzakka/recurrent-visual-attention/issues/10
        # loss_reinforce = torch.mean(-log_pi*adjusted_reward)
        loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
        loss_reinforce = torch.mean(loss_reinforce)

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce

        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return {'loss': loss,
                'acc': acc}
