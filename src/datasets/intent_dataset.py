import os
import torch
import torch.utils.data as data
from typing import Any, Callable, List, Optional, Tuple
from os.path import join as pjoin
import json


class IntentDataset(data.Dataset):
    """Prepare dataset, pad to fixed length and convert to tensor format.
    """
    def __init__(
        self,
        root: str,
        num_labels: int,
        data_name: str = '',
        data_type: str = 'train'
    ) -> None:
        if not data_name:
            data_name = 'data.json'
        self.data = json.load(open(pjoin(root, data_name)))
        self.num_labels = num_labels
        self.data_type = data_type

    def __getitem__(self, i):
        cur_data = self.data[i]
        input_ids, input_mask, input_len = convert_examples_to_features(cur_data['token_ids'])
        if self.data_type == 'train':
            label_ids = cur_data['label_ids']
            label_ids = torch.FloatTensor(label_ids)

        # tensor
        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.BoolTensor(input_mask)
        input_len = torch.LongTensor(input_len)

        if self.data_type == 'train':
            item = [input_ids, input_mask, input_len, label_ids]
        else:
            item = [input_ids, input_mask, input_len]
        

        return item

    def __len__(self):
        return int(len(self.data))


def convert_examples_to_features(input_ids, max_seq_length=128, cls_token_id=0, sep_token_id=2, pad_token_id=1):
    """Convert raw input to model input format"""

    # pad to the longest sample
    special_tokens_count = 3

    input_ids = _truncate_seq(input_ids, max_seq_length - special_tokens_count)

    input_ids = [cls_token_id] + input_ids + [sep_token_id]
    input_mask = [0] * len(input_ids)
    input_len = [len(input_ids)]

    # pad up to the max sequence length 
    padding = [pad_token_id] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask, input_len

def _truncate_seq(input_ids, max_seq_length):
    """Truncates a sequence in place to the maximum length."""

    return input_ids[:max_seq_length]