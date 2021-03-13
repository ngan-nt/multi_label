
from src.models.intent_model import LitModelIntent
from src.datasets.intent_dataset import IntentDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import csv

def collate_fn(batch):
    """Pad all sequences to longest sequence in the batch."""
    all_input_ids, all_input_mask, all_input_len = map(torch.stack, zip(*batch))
    max_len = max(all_input_len).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    return all_input_ids, all_input_mask, all_input_len

def predict(inpath, outpath):
    output = []

    # load data
    dataset = IntentDataset(inpath, num_labels=6, data_name='test.json', data_type='test')
    test_loader = DataLoader(dataset=dataset, batch_size=64, collate_fn=collate_fn)

    CKPT_PATH = 'logs/saved/last.ckpt' # check point path

    # load model from checkpoint
    trained_model = LitModelIntent.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()
    thresold = 0.5

    # inference
    for batch in tqdm(test_loader, leave=False):
        input_ids, _, _ = batch
        logits = trained_model(input_ids)
        logits[logits >= thresold] = 1
        logits[logits < thresold] = 0

        logits = logits.tolist()
        # print(logits)
        output.extend(logits)

    
    # save output model
    header = ['id', 'goal_info', 'match_info', 'match_result', 'substitution', 'penalty', 'card_info']
    with open(outpath, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header) # write header
        for i, line in enumerate(output):
            csv_writer.writerow([i] + line)


if __name__ == '__main__':
    inpath = 'data/processed_data'
    outpath = 'data/processed_data/pred_test.csv'

    predict(inpath, outpath)