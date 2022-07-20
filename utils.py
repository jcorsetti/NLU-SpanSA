import torch
import random
import numpy as np
from bert.optimization import BERTAdam

def init_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def span_overlap(seq1, seq2):
    '''
    Return True if two sequences overlap
    '''
    s1,e1 = seq1
    s2,e2 = seq2
    return not (s2 > e1 or s1 > e2)

def get_optimizer(model, config, num_train_steps):

    lr, warmup = float(config['lr']), float(config['warmup_rate'])

    if config['optimizer'] == 'bertadam':

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = BERTAdam(optimizer_grouped_parameters, lr=lr, warmup=warmup, t_total=num_train_steps)
    
    else:
        assert False, ' Optimizer {} not implemented!'.format(config['optimizer'])
    
    return optimizer

def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    return model