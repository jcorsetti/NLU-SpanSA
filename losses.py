import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax

def compute_class_loss(pred_logits, gt, mask):
  '''
  Pred_logits : [BS, N_SPAN, N_CLASS]
  gt : [BS, N_SPAN]
  mask : [BS, N_SPAN]

  '''
  BS,N_SPANS,N_CLASS = pred_logits.shape

  pred_logits = pred_logits.view(BS*N_SPANS,N_CLASS)
  gt = gt.view(BS*N_SPANS)
  mask = mask.view(BS*N_SPANS)

  losses = cross_entropy(pred_logits, gt, reduction='none')
  avg_loss = (losses * mask).sum() / mask.sum()

  return avg_loss

def imbalanced_bce(input, target):
    '''
    Version of SpanABSA modified cross-entropy, adapted for dual channel logits
    '''
    
    B,L = target.shape

    input = input.reshape(B*L, -1)
    target = target.reshape(B*L)
    
    # neg_target is used for channel 0, target for channel 1
    neg_target = 1 - target
    
    log_logits = log_softmax(input, dim=-1)
    n_pos, n_neg = torch.sum(target), torch.sum(neg_target)
    
    pos_loss = -1*torch.sum(target*log_logits[:,1]) / n_pos if n_pos > 0 else torch.tensor(0.)
    neg_loss = -1*torch.sum(neg_target*log_logits[:,0]) / n_neg if n_neg > 0 else torch.tensor(0.)

    return (pos_loss + neg_loss)

def compute_span_loss(start_logits, end_logits, gt_start, gt_end):
  
    '''
    pred_start_logits, pred_end_logits : [BS, LEN]
    gt_start, gt_end : [BS, LEN]
    '''

    # default loss
    if len(start_logits.shape) == 2:
        

        logSoftmax = torch.nn.LogSoftmax(dim=-1)
        start_logits = logSoftmax(start_logits)
        end_logits = logSoftmax(end_logits)
        
        # binary crossentropy requires probabilities instead
        start_loss = -1 * ((gt_start * start_logits).sum(-1) / gt_start.sum(-1)).mean()
        end_loss = -1 * ((gt_end * end_logits).sum(-1) / gt_end.sum(-1)).mean()
        '''

        l1loss = torch.nn.functional.l1_loss

        start_logits, end_logits = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
        start_loss = l1loss(start_logits, gt_start)
        end_loss = l1loss(end_logits, gt_end)
        '''


    else:
        
        start_loss = imbalanced_bce(start_logits, gt_start)
        end_loss = imbalanced_bce(end_logits, gt_end)

        '''
        B,L = gt_start.shape
        gt_start = gt_start.view(B*L).to(torch.long)
        gt_end = gt_end.view(B*L).to(torch.long)
        start_logits = start_logits.view(B*L,-1)
        end_logits = end_logits.view(B*L,-1)

        focal = focal_loss()
        #start_loss = cross_entropy(start_logits.view(B*L,-1), gt_start.view(B*L).to(torch.long), weight=w)
        #end_loss = cross_entropy(end_logits.view(B*L,-1), gt_end.view(B*L).to(torch.long), weight=w)
    
        start_loss = focal(start_logits, gt_start)
        end_loss = focal(end_logits, gt_end)
        '''


    return (start_loss + end_loss) / 2.

class focal_loss(nn.Module):
    '''
    Focal loss
    Lin et al., Focal Loss for Dense Object Detection, ICCV 2017
    '''

    def __init__(self, gamma=1.):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.CE_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-1.0 * logpt)
        loss = ((1-pt) ** self.gamma) * logpt
        return loss.mean()