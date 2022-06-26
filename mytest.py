import os
import torch
import numpy as np
from bert.tokenization import FullTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from my_utils import read_data, format_annotations, format_features, heuristic_multispan

PATH = 'data/absa/laptop14_train.txt'
MODEL = 'bert_models/bert-base-uncased'
DEV = 'cuda:0'
DICT = {'other': 0, 'T-NEU': 1, 'T-POS': 2, 'T-NEG': 3, 'conflict': 4}
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

tokenizer = FullTokenizer(vocab_file=os.path.join(MODEL,'vocab.txt'), do_lower_case=True)

dataset = read_data(PATH)
new_dataset = format_annotations(dataset, tag_dict=DICT, filter_empty=True)
train_features = format_features(new_dataset, tokenizer)
'''
for k in train_features[0].keys():
    print(k)

all_sample_ids = torch.tensor([f['sentence_id'] for f in train_features], dtype=torch.long)
all_sub_ids = torch.tensor([f['subtokens_id'] for f in train_features], dtype=torch.long)
all_sub_mask = torch.tensor([f['subtokens_mask'] for f in train_features], dtype=torch.long)
all_segm_ids = torch.tensor([f['segments_id'] for f in train_features], dtype=torch.long)

all_start_span = torch.tensor([f['start_span'] for f in train_features], dtype=torch.long)
all_end_span = torch.tensor([f['end_span'] for f in train_features], dtype=torch.long)
all_polarity = torch.tensor([f['polarity'] for f in train_features], dtype=torch.long)
all_polarity_mask = torch.tensor([f['polarity_mask'] for f in train_features], dtype=torch.long)

train_data = TensorDataset(all_sample_ids, all_sub_ids, all_sub_mask, all_segm_ids, all_start_span, all_end_span, all_polarity, all_polarity_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=5)

BS,SEQ_LEN,N_ASP,N_CLS = 5,96,9,5

for i, batch in enumerate(train_dataloader):

    idx, input_ids, input_mask, _, start_span, end_span, pol, pol_mask = batch

    start_pred, end_pred = torch.rand(BS,SEQ_LEN), torch.rand(BS,SEQ_LEN)
'''


start_logits = torch.zeros(30)
end_logits = torch.zeros(30)

start_logits[10] = 5.
start_logits[12] = 5.

end_logits[8] = 5.
end_logits[13] = 5.

start_logits[17] = 8.
end_logits[20] = 8.

start_logits, end_logits = start_logits.unsqueeze(0), end_logits.unsqueeze(0)

start, end, mask = heuristic_multispan(start_logits, end_logits)

