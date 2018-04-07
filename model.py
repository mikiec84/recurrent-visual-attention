from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import BaselineNet
from modules import GlimpseNet, core_network
from modules import ActionNet, LocationNet


class RecurrentAttention(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(RecurrentAttention, self).__init__()
        self.std = args.std
        rnn_hidden = args.glimpse_hidden + args.loc_hidden
        self.use_gpu = args.use_gpu
        self.num_glimpses = args.num_glimpses
        self.M = args.M

        self.glimpse_net = GlimpseNet(args.glimpse_hidden, args.loc_hidden, args.patch_size, args.num_patches, args.glimpse_scale, args.num_channels)
        self.rnn = core_network(rnn_hidden, rnn_hidden)
        self.location_net = LocationNet(rnn_hidden, 2, args.std)
        self.predictor = ActionNet(rnn_hidden, args.num_class)
        self.baseline_net = BaselineNet(rnn_hidden, 1)
        self.name = 'ram_{}_{}x{}_{}'.format(args.num_glimpses, args.patch_size, args.patch_size, args.glimpse_scale)

    def step(self, x, l_t, h_t):
        """
        @param x: image. (batch, channel, height, width)
        @param l_t: location trial. (batch, 2)
        @param h_t: last hidden state. (batch, rnn_hidden)
        @return h_t: next hidden state. (batch, rnn_hidden)
        @return l_t: next location trial. (batch, 2)
        @return b_t: baseline for step t. (batch)
        @return log_pi: probability for next location trial. (batch)
        """
        glimpse = self.glimpse_net(x, l_t)
        h_t = self.rnn(glimpse, h_t)
        mu, l_t = self.location_net(h_t)
        b_t = self.baseline_net(h_t).squeeze()

        log_pi = Normal(mu, self.std).log_prob(l_t)
        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        log_pi = torch.sum(log_pi, dim=1)

        return h_t, l_t, b_t, log_pi

    def init_loc(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        l_t = torch.Tensor(batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)
        return l_t

    def forward(self, x, y, is_training=False):
        if self.use_gpu:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        if not is_training:
            return self.forward_test(x, y)

        batch_size = x.shape[0]
        # initialize location vector and hidden state
        l_t = self.init_loc(batch_size)
        h_t = self.rnn.init_hidden(batch_size, self.use_gpu)

        # extract the glimpses
        locs = []
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses):
            # Note that log_probas is None except for t=num_glimpses-1
            h_t, l_t, b_t, p = self.step(x, l_t, h_t)

            # store
            locs.append(l_t)
            baselines.append(b_t)
            log_pi.append(p)

        log_probas = self.predictor(h_t)

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
        batch_size = x.shape[0]
        l_t = self.init_loc(batch_size)
        h_t = self.rnn.init_hidden(batch_size)

        # extract the glimpses
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses):

            # forward pass through model
            h_t, l_t, b_t, p = self.step(x, l_t, h_t)

            # store
            baselines.append(b_t)
            log_pi.append(p)

        log_probas = self.predictor(h_t)

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
        loss_reinforce = torch.mean(-log_pi*adjusted_reward)

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce

        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return {'loss': loss,
                'acc': acc}
