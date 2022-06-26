import os
import numpy as np
import torch

def read_data(path, allowed_tags=['T-NEU','T-POS','T-NEG','O']):
    
    '''
    Read a SemEval2014 file in text format.
    '''

    dataset = []
    allowed_tags = set(allowed_tags)
    with open(path, encoding='UTF-8') as f:
        # Each line is a sample
        for i,line in enumerate(f):
            tokens, tags = [], []
            # Separate annotations
            sent, annot = line.split('####')
            tagged_tokens = annot.split(' ')
            # Iterate over each annotated token
            for tagged_token in tagged_tokens:

                elems = tagged_token.split('=')
                # Sometimes '=' character is present in the sentence. This breaks the parsing
                if len(elems) == 2:
                    token, tag = elems[0], elems[1]
                else:
                    token, tag = '=', elems[-1]
                
                tag = tag.strip('\n')
                if tag in allowed_tags:
                    tags.append(tag)
                else:
                    assert False, ' tag {} not in allowed tags {}!'.format(tag, allowed_tags)
                # training with uncased BERT, all tokens are lowercased
                tokens.append(token.lower())
            
            dataset.append({
                'sentence_id' : i,
                'sentence' : sent,
                'words': tokens,
                'tags' : tags
            })

    return dataset

def format_annotations(dataset, tag_dict, empty_tag='O', filter_empty=False):
    '''
    Produce polarity and begin and end of span from a dataset
    Dataset must contain list of tokens and relative tag
    '''
    if filter_empty:
        print('Filtering sentences with no aspects.')

    new_dataset = []

    for sample in dataset:
        span_end, span_begin, span_polarity = [], [], []
        tags = sample['tags']
        previous_tag = empty_tag
        
        for idx, tag in enumerate(tags):
            # Tag changed
            if tag != previous_tag:
                # Easy case, a new entity began, no other before
                if previous_tag == empty_tag:
                    span_polarity.append(tag_dict[tag])
                    span_begin.append(idx)
                # Previous is not empty, but current is. End of object!
                elif tag == empty_tag: 
                    span_end.append(idx-1)
                # An entity ended, an other begins
                else:
                    span_end.append(idx-1)
                    span_begin.append(idx)
                    span_polarity.append(tag_dict[tag])
            previous_tag = tag

        # Special case: entity in last token of the sentence
        if previous_tag != empty_tag:
            span_end.append(idx)

        new_sample = {
            'sentence_id' : sample['sentence_id'],
            'sentence' : sample['sentence'],
            'words' : sample['words'],
            'tags' : sample['tags'],
            'span_begin' : span_begin,
            'span_end' : span_end,
            'polarity' : span_polarity
        }

        if filter_empty:
            if len(span_polarity) > 0:
                new_dataset.append(new_sample)
        else:
            new_dataset.append(new_sample)
    
    return new_dataset

def format_features(dataset, tokenizer, max_sequence_size=96) :

    MAX_SPAN = max([len(sample['polarity']) for sample in dataset])
    MAX_SEQ = max_sequence_size
    max_sent_length = 0

    new_dataset = []

    for idx_sample, sample in enumerate(dataset):
        # start position in subtoken list for each token
        token_to_subtoken = [] # equal to number of tokens
        # relative token position of each subtoken
        subtoken_to_token = [] # equal to number of subtokens
        all_subtokens = []

        # start and end position, updated to match new tokens
        start_pos = np.zeros(MAX_SPAN) 
        end_pos = np.zeros(MAX_SPAN) 
        polarity = np.zeros(MAX_SPAN)

        for i, token in enumerate(sample['words']):
            # here starts a new token
            token_to_subtoken.append(len(all_subtokens))
            subtokens = tokenizer.tokenize(token)
            
            # add new subtokens 
            all_subtokens.extend(subtokens)
            # all current subotokens belong to the i-th token
            subtoken_to_token.extend([i] * len(subtokens))

        # update sentence lenght count
        max_sent_length = max(max_sent_length,len(all_subtokens))

        if len(all_subtokens) > max_sequence_size - 2:
            print('Waning: cutting sentence {}, max sequence size is {}'.format(sample['sentence_id'], max_sequence_size))
            all_subtokens = all_subtokens[:(max_sequence_size-2)]

        for i, (start, end, pol) in enumerate(zip(sample['span_begin'], sample['span_end'], sample['polarity'])):
            # map new span start
            token_start = token_to_subtoken[start]
            # and new span end
            if end < len(sample['words']) - 1:
                token_end = token_to_subtoken[end+1] - 1
            else:
                token_end = len(all_subtokens) - 1
            
            polarity[i] = pol
            # +1 accounts for shift due to CLS tag
            start_pos[i] = token_start + 1
            end_pos[i] = token_end + 1
        
        # Mask of polarity of aspects
        polarity_mask = np.where(start_pos==0, 0, 1)

        subtoken_to_token_map = {}
        # final subtoken to token map, with +1 accounts for CLS tag added later
        for idx in range(len(all_subtokens)):
            subtoken_to_token_map[idx+1] = subtoken_to_token[idx]
        
        # add tags requested by BERT model
        all_subtokens.insert(0,'[CLS]')
        all_subtokens.append('[SEP]')

        # numpy arrays to store actual training data
        tokens_id = np.zeros(MAX_SEQ)
        tokens_mask = np.zeros(MAX_SEQ)
        # this is useless for the current task, but BERT requires it
        segment_id = np.zeros(MAX_SEQ)

        input_ids = tokenizer.convert_tokens_to_ids(all_subtokens)
        tokens_id[:len(input_ids)] = input_ids
        tokens_mask[:len(input_ids)] = 1.

        new_dataset.append({
            'sentence' : sample['sentence'],
            'sentence_id' : idx_sample,
            'tokens' : sample['words'],
            'subtokens_map' : subtoken_to_token_map,
            'subtokens' : all_subtokens,
            'subtokens_id' : tokens_id,
            'subtokens_mask' : tokens_mask,
            'segments_id' : segment_id,
            'start_span' : start_pos,
            'end_span' : end_pos,
            'polarity' : polarity,
            'polarity_mask' : polarity_mask,  
        })

    return new_dataset

def span_bound2position(span, mask, max_seq_len=96):
    '''
    span: [B, N] tensor, contains start or end indexes of up to N spans
    mask: [B] tensor, binary mask describing validity of the N spans
    
    Returns a [B, max_seq_len] binary mask tensor, with 1 at positions in which the span starts/ends.
    This is necessary to compute the cross entropy loss
    '''

    BS = mask.shape[0]
    positions = torch.zeros((BS,max_seq_len)).to(mask.device)
    batch_idx, span_idx = torch.nonzero(mask == 1, as_tuple=True)    
    positions[batch_idx, span[batch_idx, span_idx]] = 1.

    return positions

def heuristic_multispan(start_logits, end_logits, M=20, K=9, T=8.):
    '''
    THIS IS O(B*N*N!)
    start_logits, end_logits: [B,L] tensor with scores (before softmax)
    K: maximum number of aspect per sentence
    M: only top M spans are considered
    T: threshold for accepting span
    '''

    B,L = start_logits.shape
    final_start = torch.zeros(B,K)
    final_end = torch.zeros(B,K)
    mask = torch.zeros(B,K)
    # must iterate over batches
    
    for i_b in range(B):
        cur_start, cur_end = start_logits[i_b], end_logits[i_b]
        # get topK candidates
        best_start = torch.argsort(cur_start, descending=True)[:M]
        best_end = torch.argsort(cur_end, descending=True)[:M]
        # list of scores and accepted spans 
        score, filtered = [], []
        
        # iterate over possible spans
        for i_s in best_start:
            for i_e in best_end:
                # calculate score and lenght
                cur_score = cur_start[i_s] + cur_end[i_e]
                cur_lenght = i_e - i_s + 1
                
                # score must be of a certain threshold, and end must be before start
                if i_s <= i_e and cur_score >= T:
                    score.append(cur_score.item() - cur_lenght.item())
                    filtered.append((i_s.item(), i_e.item()))
                    print('Accepted span ({},{}) with score {}!'.format(i_s,i_e,cur_score-cur_lenght))

        accepted = 0
        # NMS part!

        while (filtered != []) and (accepted < K):
            # get best span, remove score and index
            best_idx = score.index(max(score))
            best_span = filtered[best_idx]
            score.pop(best_idx)
            filtered.pop(best_idx)

            # update final span list
            final_start[i_b, accepted] = best_span[0]
            final_end[i_b, accepted] = best_span[1]
            mask[i_b, accepted] = 1
            accepted += 1

            # removing spans overlapping with the best one found
            tmp_score, tmp_filtered = [], []
            for i, span in enumerate(filtered):
                if not span_overlap(best_span, span):
                    tmp_score.append(score[i])
                    tmp_filtered.append(span)
            filtered, score = tmp_filtered, tmp_score
    
    return final_start, final_end, mask

def span_overlap(seq1, seq2):
    '''
    Return True if two sequences overlap
    '''
    s1,e1 = seq1
    s2,e2 = seq2
    return not (s2 > e1 or s1 > e2)
