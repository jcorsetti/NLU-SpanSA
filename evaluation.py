import torch
import numpy as np
from utils import span_overlap

def heuristic_multispan(start_logits, end_logits, sequence_mask, M=20, K=9, T=8.):
    '''
    THIS IS O(B*N*N!)
    start_logits, end_logits: [B,L] tensor with scores (before softmax)
    sequence_mask: mask tensor [B,L], used to filter out spans occurring outside actual sentence lenght
    K: maximum number of aspect per sentence
    M: only top M spans are considered
    T: threshold for accepting span
    '''

    B,L = start_logits.shape
    
    final_start = torch.zeros((B,K))
    final_end = torch.zeros((B,K))
    mask = torch.zeros((B,K))
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
                
                # invalid spans: start or end occurs at an index where sentence is not present!
                if sequence_mask[i_b,i_s] == 0. or sequence_mask[i_b,i_e] == 0.:
                    continue

                # score must be of a certain threshold, and end must be before start
                if i_s <= i_e and cur_score >= T:
                    score.append(cur_score.item() - cur_lenght.item())
                    filtered.append((i_s.item(), i_e.item()))
                    #print('Accepted span ({},{}) with score {}!'.format(i_s,i_e,cur_score-cur_lenght))

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

def evaluate(model, eval_dataloader, eval_features, pol_dict, config, device):
    '''
    Evaluates the full model on a provided dataset
    '''

    model.eval()

    MAX_PROP_SPANS = int(config['heuristics']['max_prop_spans'])
    MAX_ACCEPTED_SPANS = int(config['heuristics']['max_accepted_spans'])
    LOG_T =  float(config['heuristics']['logits_threshold'])

    preds = {}

    all_common, all_retrieved, all_relevant = 0., 0., 0.
    with torch.no_grad():

        for step, batch in enumerate(eval_dataloader):

            batch = tuple(t.to(device) for t in batch)  
            input_ids, input_mask, segment_ids, start_span, end_span, polarity, polarity_mask, example_indices = batch

            # predict span logits
            start_logits, end_logits = model.extract_span(input_ids, input_mask, segment_ids)
            # run euristic to extract span indexes
            final_start, final_end, span_mask = heuristic_multispan(start_logits, end_logits, input_mask, MAX_PROP_SPANS, MAX_ACCEPTED_SPANS, LOG_T)
            final_start, final_end, span_mask = final_start.to(device).to(torch.long), final_end.to(device).to(torch.long), span_mask.to(device).to(torch.long)

            # classify predicted spans
            polarity_logits = model.classify_span(input_ids, input_mask, segment_ids, final_start, final_end)
            
            # predicted class from predicted spans
            polarity_pred_class = torch.argmax(torch.softmax(polarity_logits,dim=-1),dim=-1)
            
            common, retrieved, relevant, cur_preds = evaluate_batch(final_start, final_end, polarity_pred_class, span_mask, example_indices, eval_features, pol_dict)
            
            all_common += common
            all_retrieved += retrieved
            all_relevant += relevant
            preds.update(cur_preds)

    if all_retrieved > 0:
      p = all_common / all_retrieved
    else:
      p = 0.
    
    r = all_common / all_relevant
    if p+r > 0:
        f1 = (2*p*r) / (p+r)
    else:
        f1 = 0.

    return p, r, f1, all_common, all_retrieved, all_relevant, preds

def evaluate_batch(pred_start, pred_end, pred_polarity, span_mask, indexes, features, pol_dict):

    BS = pred_start.shape[0]
    retrieved, relevant, common = 0., 0., 0.

    pred_start = pred_start.detach().cpu()
    pred_end = pred_end.detach().cpu()
    pred_polarity = pred_polarity.detach().cpu()
    span_mask = span_mask.detach().cpu()

    preds = {}
    reverse_dict = {v:k for k,v in pol_dict.items()}

    for i in range(BS):

        start_i, end_i, pol_i, mask_i, idx = pred_start[i], pred_end[i], pred_polarity[i], span_mask[i], indexes[i]
        gt = features[idx.item()]

        predicted_span = []
        predicted_polarity = []
        predicted_text = []

        # iterate over predicted spans
        for start, end, pol, mask in zip(start_i, end_i, pol_i, mask_i):
          
            # use mask to get whether this is a valid prediction
            if mask == 1:
                predicted_span.append((start.item(), end.item()))
                predicted_polarity.append(pol)
                
                # looking in ground truth to see if I found this span and polarity
                for gt_start, gt_end, gt_pol in zip(gt['start_span'], gt['end_span'], gt['polarity']):

                    # Span correct!
                    if (gt_start == start) and (gt_end == end):

                        if (gt_pol == pol):
                            common += 1.
                        
                        text_span = gt['subtokens'][int(gt_start.item()):int(gt_end.item())+1]
                        stripped_chars = [char.replace("##", "") for char in text_span]
                        predicted_text.append({
                            'sentence' : gt['sentence'],
                            'text' : ' '.join(stripped_chars),
                            'pred_cls' : reverse_dict[int(pol.item())],
                            'gt_cls' : reverse_dict[int(gt_pol.item())]

                        })

                        # correspondence has been found: exit from cycle
                        break
        
        preds[idx.item()] = predicted_text
            
        true_terms = np.count_nonzero(gt['polarity_mask'])
        predicted_terms = np.count_nonzero(mask_i)
        relevant += true_terms
        retrieved += predicted_terms

    
    return common, retrieved, relevant, preds

def evaluate_classonly(model, eval_dataloader, eval_features, config, device):
    '''
    Evaluates only the aspect classification branch of a model on a specific dataset. Ground truth spans are used
    '''

    model.eval()

    confusion_matrix = np.zeros((5,5))

    with torch.no_grad():

        for step, batch in enumerate(eval_dataloader):

            batch = tuple(t.to(device) for t in batch)  
            input_ids, input_mask, segment_ids, start_span, end_span, polarity, polarity_mask, example_indices = batch

            # classify spans
            polarity_logits = model.classify_span(input_ids, input_mask, segment_ids, start_span, end_span)
            
            # predicted class from spans
            polarity_pred_class = torch.argmax(torch.softmax(polarity_logits,dim=-1),dim=-1)
            
            cur_mat = evaluate_batch_classonly(polarity_pred_class, polarity_mask, example_indices, eval_features)
            
            confusion_matrix += cur_mat
            

    correct = 0.
    total = np.sum(confusion_matrix)
    # sum of principal diagonal is equal to the number of correct classes
    for i in range(confusion_matrix.shape[0]):
        correct += confusion_matrix[i,i]
    
    accuracy = correct / total

    return accuracy, correct, total

def evaluate_batch_classonly(pred_polarity, span_mask, indexes, features):

    BS = pred_polarity.shape[0]

    pred_polarity = pred_polarity.detach().cpu()
    span_mask = span_mask.detach().cpu()

    confusion_matrix = np.zeros((5,5))

    for i in range(BS):

        pol_i, mask_i, idx = pred_polarity[i], span_mask[i], indexes[i]
        gt = features[idx.item()]

        # iterate over class couples
        for pred_pol, gt_pol, mask in zip(pol_i, gt['polarity'], mask_i):
          
            # use mask to get whether this is a valid prediction
            if mask == 1:
            
                confusion_matrix[int(gt_pol.item()),int(pred_pol.item())] += 1
                
    return confusion_matrix

def evaluate_spanonly(model, eval_dataloader, eval_features, config, device):
    
    model.eval()

    MAX_PROP_SPANS = int(config['heuristics']['max_prop_spans'])
    MAX_ACCEPTED_SPANS = int(config['heuristics']['max_accepted_spans'])
    LOG_T =  float(config['heuristics']['logits_threshold'])

    all_common, all_retrieved, all_relevant = 0., 0., 0.

    with torch.no_grad():

        for step, batch in enumerate(eval_dataloader):

            batch = tuple(t.to(device) for t in batch)  
            input_ids, input_mask, segment_ids, start_span, end_span, polarity, polarity_mask, example_indices = batch

            # predict span logits
            start_logits, end_logits = model.extract_span(input_ids, input_mask, segment_ids)
            # run euristic to extract span indexes
            final_start, final_end, span_mask = heuristic_multispan(start_logits, end_logits, input_mask, MAX_PROP_SPANS, MAX_ACCEPTED_SPANS, LOG_T)
            final_start, final_end, span_mask = final_start.to(device).to(torch.long), final_end.to(device).to(torch.long), span_mask.to(device).to(torch.long)

            common, retrieved, relevant = evaluate_batch_spanonly(final_start, final_end, span_mask, example_indices, eval_features)
            
            all_common += common
            all_retrieved += retrieved
            all_relevant += relevant

    if all_retrieved > 0:
      p = all_common / all_retrieved
    else:
      p = 0.
    
    r = all_common / all_relevant
    if p+r > 0:
        f1 = (2*p*r) / (p+r)
    else:
        f1 = 0.

    return p, r, f1
    

def evaluate_batch_spanonly(pred_start, pred_end, span_mask, indexes, features):

    BS = pred_start.shape[0]
    retrieved, relevant, common = 0., 0., 0.

    pred_start = pred_start.detach().cpu()
    pred_end = pred_end.detach().cpu()
    span_mask = span_mask.detach().cpu()

    for i in range(BS):

        start_i, end_i, mask_i, idx = pred_start[i], pred_end[i], span_mask[i], indexes[i]
        gt = features[idx.item()]

        predicted_span = []

        # iterate over predicted spans
        for start, end, mask in zip(start_i, end_i, mask_i):
          
            # use mask to get whether this is a valid prediction
            if mask == 1:
                predicted_span.append((start.item(), end.item()))
                
                # looking in ground truth to see if I found this span and polarity
                for gt_start, gt_end in zip(gt['start_span'], gt['end_span']):

                    # Span correct!
                    if (gt_start == start) and (gt_end == end):

                        # correspondence has been found: exit from cycle
                        common += 1.
                        break
            
        true_terms = np.count_nonzero(gt['polarity_mask'])
        predicted_terms = np.count_nonzero(mask_i)
        relevant += true_terms
        retrieved += predicted_terms

    return common, retrieved, relevant