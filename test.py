import sys
import os
import json
import numpy as np

import bert.tokenization as tokenization
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features

model = 'bert_models/bert-base-uncased'
data_dir = 'data/absa'
train_file='laptop14_train.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model,'vocab.txt'), do_lower_case=True)

train_path = os.path.join(data_dir, train_file)
train_set = read_absa_data(train_path)
train_examples = convert_absa_data(dataset=train_set)
train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=96)

i = 17
print(train_examples[i].sent_tokens)
print(train_examples[i].polarities)
print(train_examples[i].term_texts)

print(train_features[i].segment_ids)
print('Tokens and map to original')
print(train_features[i].tokens)
print(train_features[i].input_ids)
print(train_features[i].token_to_orig_map)

print('Start and end position of aspects')
print(train_features[i].start_positions)
print(train_features[i].end_positions)

print(train_features[i].start_indexes)
print(train_features[i].end_indexes)

print('Polarity position, labels with masks')
print(train_features[i].polarity_positions)
print(train_features[i].polarity_labels)
print(train_features[i].label_masks)        


