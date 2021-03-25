import numpy as np
import Levenshtein

import torch

class Dataset:
    def __init__(self, query, base, nearest_info, distance_info, alphabet_table, max_str_len, neighbors_num=100):
        self.query = query
        self.base = base

        self.nearest_info = nearest_info
        self.distance_info = distance_info

        self.neighbors_num = neighbors_num

        self.alphabet_table = alphabet_table
        self.alphabet_len = len(alphabet_table)
        self.max_str_len = max_str_len

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        anchor_string = self.query[idx]

        positive_string, negative_string, positive_distance, negative_distance = \
                self._get_pos_neg_string(idx)

        positive_distance = torch.tensor(positive_distance, dtype=torch.float)
        negative_distance = torch.tensor(negative_distance, dtype=torch.float)

        anchor_onehot_string = self._encoding_string(anchor_string)
        positive_onehot_string = self._encoding_string(positive_string)
        negative_onehot_string = self._encoding_string(negative_string)

        return (anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance)

    def _get_pos_neg_string(self, idx):
        positive_idx = np.random.randint(self.neighbors_num-1)
        negative_idx = np.random.randint(positive_idx+1, self.neighbors_num)

        positive_idx = self.nearest_info[idx, positive_idx]
        negative_idx = self.nearest_info[idx, negative_idx]

        positive_string = self.base[positive_idx]
        negative_string = self.base[negative_idx]

        positive_distance = self.distance_info[idx, positive_idx]
        negative_distance = self.distance_info[idx, negative_idx]

        return positive_string, negative_string, positive_distance, negative_distance


    def _encoding_string(self, string):
        onehot_string = torch.zeros((self.alphabet_len, self.max_str_len))

        for i, char in enumerate(string):
            char_idx = self.alphabet_table.index(char)
            onehot_string[char_idx][i] = 1

        return onehot_string

class SplitDataset:
    def __init__(self, path_to_dataset, training_set_num, query_set_num):
        data = self._load_dataset(path_to_dataset)

        self.alphabet_table = self._get_alphabet_table(data)
        self.max_str_len = self._get_max_str_len(data)

        self.train_data, self.query_data, self.base_data = \
                self._split_data(data, training_set_num, query_set_num)

        self.train_nearest_info, self.train_distance_info = \
                self._get_data_info(self.train_data, self.train_data)
        self.eval_nearest_info, self.eval_distance_info= \
                self._get_data_info(self.query_data, self.base_data)

    def get_alphabet_table(self):
        return self.alphabet_table

    def get_max_str_len(self):
        return self.max_str_len

    def get_train_data(self):
        return self.train_data, self.train_nearest_info, self.train_distance_info

    def get_eval_data(self):
        return self.query_data, self.base_data, self.eval_nearest_info, self.eval_distance_info

    def _get_alphabet_table(self, data):
        total_string = "".join(s for s in data)
        alphabet_table = list(set(total_string))
        return alphabet_table

    def _get_max_str_len(self, data):
        len_list = [len(s) for s in data]
        return max(len_list)

    def _load_dataset(self, path_to_dataset):
        lines = open(path_to_dataset, "rb").read().splitlines()
        return [line.decode("utf8", "ignore") for line in lines]

    def _split_data(self, data, training_set_num, query_set_num):
        """
        return (list) : list of string
        """
        data_len = len(data)
        data_idx = np.arange(data_len)

        np.random.shuffle(data_idx)

        train_data = [data[idx] for idx in data_idx[:training_set_num]]
        query_data = [data[idx] for idx in data_idx[training_set_num:query_set_num+training_set_num]]
        base_data = [data[idx] for idx in data_idx[query_set_num:]]

        return train_data, query_data, base_data


    def _get_data_info(self, query, base):
        distance_info = self._get_distance_info(query, base)
        nearest_info = self._get_nearest_info(distance_info)
        return nearest_info, distance_info


    def _get_nearest_info(self, distance_info):
        """
        return (ndarray) : A nearest index metric with shape (len(query), len(base))
        """
        nearest_info = np.zeros_like(distance_info, dtype=np.int)

        for r in range(nearest_info.shape[0]):
            nearest_info[r, :] = np.argsort(distance_info[r])

        return nearest_info


    def _get_distance_info(self, query, base):
        """
        return (ndarray) : A distance metric with shape (len(query), len(base))
        """
        query_len = len(query)
        base_len = len(base)

        dist = np.zeros((query_len, base_len))
        for q_idx in range(query_len):
            for b_idx in range(base_len):
                dist[q_idx, b_idx] = Levenshtein.distance(query[q_idx], base[b_idx])

        return dist
