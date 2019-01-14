"""
Name:           nnLSTM.py
Function:       Provide the LSTM neural network model.

Author:         LiXu
Create Date:    2019.01.13 09:00
Modify Date:    2019.01.12
Version:        0.1
"""

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class TwoBiLSTM(nn.Module):

    def __init__(self, args):
        super(TwoBiLSTM, self).__init__()

        self.args = args

        # Define the lstm model of the target
        self.target_hidden_dim = args.target_hidden_dim
        self.target_num_layers = args.target_num_layers

        V_target = args.target_embed_num       # Number of the target vocabulary.
                                               # Get from the target dataset vocabulary.
        D_target = args.embed_dim
        C_target = args.FIVE_CLASS_TASK

        self.target_embed = nn.Embedding(V_target, D_target)
        self.target_bilstm = nn.LSTM(D_target,
                                     self.target_hidden_dim // 2,
                                     num_layers=1,
                                     # dropout=args.dropout,
                                     bidirectional=True)

        # self.target_linear = nn.Linear(self.target_hidden_dim, C_target)

        # Define the lstm model of the tweet.
        self.tweet_hidden_dim = args.tweet_hidden_dim
        self.tweet_num_layers = args.tweet_num_layers

        V_tweet = args.tweet_embed_num       # Number of the tweet vocabulary.
                                             # Get from the tweet dataset vocabulary.
        D_tweet = args.embed_dim
        C_tweet = 3

        self.tweet_embed = nn.Embedding(V_tweet, D_tweet, padding_idx=1)
        self.tweet_bilstm = nn.LSTM(D_tweet,
                                    self.tweet_hidden_dim // 2,
                                    num_layers=1,
                                    # dropout=args.dropout,
                                    bidirectional=True)

        self.tweet_linear2 = nn.Linear(self.tweet_hidden_dim, self.tweet_hidden_dim // 2)
        self.tweet_linear1 = nn.Linear(self.tweet_hidden_dim // 2, C_tweet)

        # Define the softmax linear concatenate layer.
        # self.softmax_cat = nn.Linear(C_tweet, C_tweet)

    def forward(self, target_input, tweet_input):

        # Compute the target lstm output firstly.
        embed_tar = self.target_embed(target_input)
        uniword_input = embed_tar.view(1, embed_tar.size(0), embed_tar.size(1))

        target_output, (target_hn, target_cn) = self.target_bilstm(uniword_input)

        # h_target = self.target_linear(target_output)

        # Then Compute the tweet lstm model.
        embed_twe = self.tweet_embed(tweet_input)

        hn_initial = torch.zeros(target_cn.size())
        tweet_output, (tweet_hn, tweet_cn) = self.tweet_bilstm(embed_twe, (hn_initial, target_cn))

        # h_tweet = F.tanh(self.tweet_linear2(tweet_output))
        # h_tweet = self.tweet_linear1(h_tweet)

        output = torch.cat([target_output, tweet_output], 0)

        # Concatenate the hidden output of target and tweet.
        # And cross the full connected neural network.

        # TODO: Not concatenate the output of lstm?
        # target_tweet = torch.cat([target_hn, tweet_hn], 2)

        # output = self.softmax_cat(h_target + h_tweet)
        # output = F.softmax(F.tanh(output))

        # output = F.tanh(output)
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        output = F.max_pool1d(output, output.size(2))
        output = output.squeeze(2)

        output = self.tweet_linear2(output)
        output = torch.tanh(output)

        output = self.tweet_linear1(output)
        output = F.softmax(F.tanh(output), dim=1)

        return output



