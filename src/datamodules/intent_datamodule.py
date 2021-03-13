from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch
from .tokenizer import XLMRobertaTokenizer
from os.path import join as pjoin
import json
from src.datasets.intent_dataset import IntentDataset


class IntentDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.train_val_test_split = kwargs["train_val_test_split"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]
        self.num_labels = kwargs["num_labels"]

        self.tokenizer = XLMRobertaTokenizer(kwargs["tokenizer_dir"]) # load tokenizer

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.dataset = IntentDataset(self.data_dir, self.num_labels, data_name='train.json')

        # split train val test
        num_train = int(len(self.dataset)*self.train_val_test_split[0])
        num_val = int(len(self.dataset)*self.train_val_test_split[1])
        num_test = len(self.dataset) - num_train - num_val
        self.train_val_test_split_num = [num_train, num_val, num_test]

        self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.train_val_test_split_num)
         

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)
 
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)    


def collate_fn(batch):
    """Pad all sequences to longest sequence in the batch."""
    all_input_ids, all_input_mask, all_input_len, all_label_ids = map(torch.stack, zip(*batch))
    max_len = max(all_input_len).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    return all_input_ids, all_input_mask, all_input_len, all_label_ids