# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the Vocab class for converting string to id and back
"""

import numpy as np


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, tokens, embeddings):
        assert embeddings.shape[0] == len(tokens)
        self.initial_tokens = tokens
        self.id2token = dict(zip(range(len(tokens)), tokens ))
        self.token2id = dict(zip(tokens, range(len(tokens))))
        self.token_cnt = dict(zip(tokens, [1]*len(tokens)))
        self.embed_dim = embeddings.shape[1]
        self.embeddings = embeddings

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def count(self, tokens, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        for token in tokens:
            if cnt > 0:
                if token in self.token_cnt:
                    self.token_cnt[token] += cnt

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [token for token in self.initial_tokens if
                           self.token_cnt[token] >= min_cnt or token in [self.pad_token, self.unk_token]]
        filtered_ids = [idx for idx, token in enumerate(self.initial_tokens) if
                        self.token_cnt[token] >= min_cnt or token in [self.pad_token, self.unk_token]]
        # rebuild the token x id map
        self.embeddings = self.embeddings[filtered_ids]
        self.token2id = dict(zip(filtered_tokens, range(len(filtered_tokens))))
        self.id2token = dict(zip(range(len(filtered_tokens)), filtered_tokens))

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def set_embeddings(self, embeddings):
        assert len(self.initial_tokens) == self.embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        self.embeddings = embeddings

    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
