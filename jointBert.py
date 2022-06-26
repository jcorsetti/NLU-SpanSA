from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from bert.modeling import BertModel, BERTLayerNorm
from torch.nn.functional import softmax

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output


class SpanStandardHead(nn.Module):
    def __init__(self, config):
        super(SpanStandardHead, self).__init__()
        self.span_head = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence):

        logits = self.span_head(sequence)   # [N, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        return start_logits, end_logits
        
class PolClassStandardHead(nn.Module):
    def __init__(self, config):
        super(PolClassStandardHead, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, 1)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.polarity_class = nn.Linear(config.hidden_size, 5)


    def forward(self, sequence, span_starts, span_ends, mask):
        
        BS = span_starts.shape[0]

        span_output, span_mask = get_span_representation(span_starts, span_ends, sequence, mask)  # [N*M, JR, D], [N*M, JR]
        
        N_SPANS = span_output.shape[0] // BS

        span_score = self.fc1(span_output)
        span_score = span_score.squeeze(-1)  # [N*M, JR]
        span_feats = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]
        
        span_feats = self.fc2(span_feats)
        span_feats = self.act(span_feats)
        span_feats = self.dropout(span_feats)
        polarity_logits = self.polarity_class(span_feats)  # [N*M, 5]
        polarity_logits = softmax(polarity_logits,dim=-1)

        return polarity_logits.view(BS,N_SPANS,-1)

class JointBert(nn.Module):
    def __init__(self, config, span_head_arch=None, pol_head_arch=None):
        super(JointBert, self).__init__()
        
        self.bert = BertModel(config)
        if span_head_arch is None:
            self.span_head = SpanStandardHead(config)
        else:
            assert False, ' arch {} not implemented for Span Head!'.format(span_head_arch)
        if pol_head_arch is None:
            self.pol_head = PolClassStandardHead(config)
        else:
            assert False, ' arch {} not implemented for Polarity Classification Head!'.format(pol_head_arch)
        

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, mask, token_type_ids, span_starts, span_ends):

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, mask)
        sequence_output = all_encoder_layers[-1]

        span_start_logits, span_end_logits = self.span_head(sequence_output)

        polarity_logits = self.pol_head(sequence_output, span_starts, span_ends, mask)

        return span_start_logits, span_end_logits, polarity_logits
