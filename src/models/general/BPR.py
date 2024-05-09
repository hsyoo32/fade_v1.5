# -*- coding: UTF-8 -*-
import torch.nn as nn
from models.Model import Model

class BPR(Model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return Model.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, u_ids, i_ids, flag):
        # print('u_ids:', u_ids)
        # print('i_ids:', i_ids)
        #print('u_ids shape:', u_ids.shape)
        #print('i_ids shape:', i_ids.shape)

        # self.check_list = []
        u_ids = u_ids.repeat((1, i_ids.shape[1]))  # [batch_size, -1]
        #print('u_ids shape:', u_ids.shape)

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
        #print('prediction shape:', prediction.shape)
            
        #return prediction.view(len(u_ids), -1)
        return prediction

    def model_(self, user, items, flag):
        user = user.repeat((1, items.shape[0])).squeeze(0)

        cf_u_vectors = self.u_embeddings(user)
        cf_i_vectors = self.i_embeddings(items)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
            
        return prediction


