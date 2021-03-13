import sys
sys.path.append('')
import json
from src.datamodules.tokenizer import XLMRobertaTokenizer
from nltk import word_tokenize
from tqdm import tqdm
import csv

def format_data_input(inpath, outpath, data_type='train'):
    output = [] # store output here
    tokenizer = XLMRobertaTokenizer('lib/envibert') # tokenizer

    # read data
    with open(inpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip headers
        for row in reader:
            # tokenize text
            if data_type == 'train':
                text = row[0]
            else:
                text = row[1]
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            sample = {
                'text': text,
                'token_ids': token_ids,
            }
            if data_type == 'train':
                label_ids = row[1:]
                label_ids = [int(l) for l in label_ids] # convert string to int
                sample['label_ids'] = label_ids


            output.append(sample)
    
    # write output
    json.dump(output, open(outpath, 'w'), ensure_ascii=False)


if __name__ == '__main__':

    ## FOR TRAIN DATA
    inpath = 'data/raw_data/train.csv' # input data path
    outpath = 'data/processed_data/train.json' # processed data path

    format_data_input(inpath, outpath)

    ## FOR TEST DATA
    inpath = 'data/raw_data/test.csv' # input data path
    outpath = 'data/processed_data/test.json' # processed data path

    format_data_input(inpath, outpath, data_type='test')