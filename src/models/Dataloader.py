import torch
import logging
import os
import numpy as np
import copy
from random import randint
from torch.utils.data import Dataset as BaseDataset
from utils import utils


class Dataset(BaseDataset):
    def __init__(self, model, args, corpus, data_idx):
        self.model = model  # model object reference
        self.corpus = corpus  # reader object reference
        self.train_ratio = args.train_ratio
        self.mini_batch_path = corpus.mini_batch_path
    

        self.batch_size = args.batch_size
        self.train_boundary = corpus.n_train_batches
        self.snapshots_path = corpus.snapshots_path
        self.n_snapshots = corpus.n_snapshots

        if args.dyn_method == 'fulltrain':
            self.train_file = os.path.join(corpus.snapshots_path, 'hist_block'+str(data_idx))
        elif args.dyn_method == 'finetune':
            self.train_file = os.path.join(corpus.snapshots_path, 'incre_block'+str(data_idx))
        
        self.train_data = utils.read_data_from_file_int(self.train_file)
        self.train_data = np.array(self.train_data)

        print(self.train_data.shape)
        

        user_attr = utils.read_data_from_file_int(corpus.user_attr_path)
        self.user_attr_dict = {}
        for user in user_attr:
            self.user_attr_dict[user[0]] = user[1] # gender M: 1, F: 0
        self.DRM = args.DRM

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index: int) -> dict:
        current = self._get_feed_dict(index)
        #print('index: {}'.format(index))
        return current

    def _get_feed_dict(self, index: int) -> dict:


        user_id, item_id = self.train_data[index]
        neg_items = self._sample_neg_items(user_id).squeeze()
        user_id, item_id = torch.tensor([user_id]), torch.tensor([item_id])
        item_id_ = torch.cat((item_id, neg_items), axis=-1)
        
        #item_id_ = torch.cat((item_id.reshape(-1, 1), neg_items), axis=-1)
        feed_dict = {'user_id': user_id, #(batch_size, )
                        'item_id': item_id_} #(batch_size, 1+neg_items)

        sen_attr = []
        for user in user_id:
            sen_attr.append(self.user_attr_dict[user.item()])
        sen_attr = torch.from_numpy(np.array(sen_attr))
        feed_dict['attr'] = sen_attr

        return feed_dict

    def _sample_neg_items(self, user_id):
        #num_neg = self.model.num_neg
        num_neg = max(self.model.num_neg, self.model.num_neg_fair)

        neg_items = torch.zeros(size=(1, num_neg), dtype=torch.int64)
        #neg_items = torch.zeros(size=(num_neg), dtype=torch.int64)

        #for idx, user in enumerate(self.corpus.user_list[index:index_end]): # Automatic coverage?
        #for idx, user in enumerate(user_id): # Automatic coverage?

        user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user_id])
        # By copying, it may not collide with other process with same user index
        for neg in range(num_neg):
            neg_item = self._randint_w_exclude(user_clicked_set)
            neg_items[0][neg] = neg_item
            # Skip below: one neg for train
            user_clicked_set = np.append(user_clicked_set, neg_item)

        return neg_items

    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.corpus.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem