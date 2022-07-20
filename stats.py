import numpy as np
from sys import argv
from dataset import read_data, format_annotations

# script used to calculate the dataset statistics

DICT = {'OTHER': 0, 'T-NEU': 1, 'T-POS': 2, 'T-NEG': 3, 'CONF' : 4}

file = 'data/absa/laptop14_{}.txt'.format(argv[1])

raw_dataset = read_data(file)

processed_dataset = format_annotations(raw_dataset, DICT, filter_empty=False)

n_sents = len(processed_dataset)
num_tokens = 0
empty_sents = 0
avg_aspect_len = 0.
num_aspect = 0
neu_aspect, pos_aspect, neg_aspect = 0,0,0

for sent_data in processed_dataset:
    num_tokens += len(sent_data['words'])
    begin, end, pol = np.asarray(sent_data['span_begin']),np.asarray(sent_data['span_end']),np.asarray(sent_data['polarity'])
    num_aspect += len(sent_data['polarity'])
    neu_aspect += np.count_nonzero(pol==1)
    pos_aspect += np.count_nonzero(pol==2)
    neg_aspect += np.count_nonzero(pol==3)

    if len(sent_data['polarity']) == 0:
        empty_sents += 1
    else:
        lengths = end - begin + 1
        avg_aspect_len += sum(lengths)

print('Num sentences: {}'.format(n_sents))
print('  of which empty: {}'.format(empty_sents))
print('  avg length in tokens: {:.1f}'.format(float(num_tokens)/float(n_sents)))

print('Num aspects: {}'.format(num_aspect))
print('  neutral: {}'.format(neu_aspect))
print('  negative: {}'.format(neg_aspect))
print('  positive: {}'.format(pos_aspect))
print('  avg per nonempty sentence: {:.1f}'.format(float(num_aspect)/float(n_sents-empty_sents)))





