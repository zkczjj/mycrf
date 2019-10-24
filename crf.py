import torch
import torch.nn as nn

inf = 10000.0

class CRF(nn.Module):
    
    def __init__(self, tag_scheme='BIOES'):
        super(CRF, self).__init__()

        self.tag_scheme = tag_scheme
        self.label_vocab = list(tag_scheme)
        self.n_labels = len(self.label_vocab) + 2
        self.start = self.n_labels - 2
        self.end = self.n_labels - 1
        
        transition = torch.ones(self.n_labels, self.n_labels)
        self.transition = nn.Parameter(transition)
        self.reset_parameter()

    def reset_parameter(self):
        ''' Init transtion matrix
            Example:
                      START B I O E S END
                START     X
                    B     X X   X   X
                    I     X X   X   X
                    O     X   X   X
                    E     X   X   X
                    S     X   X   X
                  END     X X X X X X X  
        '''

        self.transition.data[:, self.start] = -inf
        self.transition.data[self.end, :] = -inf

        for label_from_idx, label_from in enumerate(self.label_vocab):
            for label_to_idx, label_to in enumerate(self.label_vocab):
                is_allowed = any([label_from in ['O', 'E', 'S']and label_to in ['O', 'B', 'S'],
                                  label_from in ['B', 'I'] and label_to in ['I', 'E']])
                if not is_allowed:
                    self.transition.data[label_from_idx, label_to_idx] = -inf

    def log_sum_exp(self, tensor, dim=0, keepdim=False):
        '''
            Trick: select max value, reduce complexity
        '''
        m, _ = tensor.max(dim, keepdim=keepdim)
        stable_vec = tensor - m if keepdim else tensor - m.unsqueeze(dim)
        result = m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()
        return result

    def sequence_mask(self, lens, max_len=None):
        ''' Get mask to distinguish pad
            Para:
                lens: sequence lengths, batch_size
                max_len: max sequence lengths
            Return: mask
        '''
        batch_size = lens.shape[0]
        if max_len is None:
            max_len = lens.max().item()
        ranges = torch.arange(0, max_len).long()
        ranges_exp = ranges.unsqueeze(0).expand(batch_size, max_len)
        lens_exp = lens.unsqueeze(1).expand_as(ranges_exp)
        mask = ranges_exp < lens_exp
        return mask

    def pad_logits(self, logits):
        ''' Pad the linear layer output with <START> and <END> scores.
            Param:
                logits: Linear layer output (no non-linear function).
                        shape = (batch_size, seq_len, n_labels)
            Return: (batch_size, seq_len, n_labels+2)
        '''
        batch_size, seq_len, n_labels = logits.shape
        pads = logits.new_full((batch_size, seq_len, 2), -inf, requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def get_trans_score(self, labels, lens):
        ''' Caculate transtion score
            Para:
                labels: (batch_size, seq_lens)
                lens:   (batch_size,)
            Return: trans score vec
        '''
        batch_size, seq_len = labels.shape
        mask = self.sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        #1.1 from start
        labels_new = labels.new_empty((batch_size, seq_len + 2))
        labels_new[:, 0] = self.start
        labels_new[:, 1:-1] = labels
        #1.2 to end
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        #1.3 add
        labels = (1 - mask) * pad_stop + mask * labels_new
        
        trans = self.transition
        trans_exp = trans.unsqueeze(0).expand(batch_size, self.n_labels, self.n_labels)

        #2.1 from, from -> to
        from_ = labels[:, :-1]
        from_exp = from_.unsqueeze(-1).expand(*from_.shape, self.n_labels)
        from_trans = torch.gather(trans_exp, dim=1, index=from_exp)
        #2.2 to, from -> to
        to_exp = labels[:, 1:].unsqueeze(-1)
        trans_score = torch.gather(from_trans, dim=-1, index=to_exp).squeeze(-1)
        mask = self.sequence_mask(lens + 1).float()
        return trans_score * mask

    def get_state_score(self, logits, labels, lens):
        ''' Caculate state score
            Para:
                logits: (batch_size, seq_lens, n_labels)
                labels: (batch_size, seq_lens)
                lens:   (batch_size,)
            Return: state score vec
        '''
        labels = labels.unsqueeze(-1)
        mask = self.sequence_mask(lens).float()
        scores = torch.gather(logits, dim=2, index=labels).squeeze(-1)
        return scores * mask

    def get_gold_score(self, logits, labels, lens):
        ''' Caculate real_path score
            Para:
                logits: state score vec (batch_size, seq_lens, n_labels)
                labels: (batch_size, seq_lens)
                lens: (batch_size,)
            Return: real path score vec
        '''
        trans_score = self.get_trans_score(labels, lens).sum(dim=1).squeeze(-1)
        state_score = self.get_state_score(logits, labels, lens).sum(dim=1).squeeze(-1)
        return trans_score + state_score

    def get_all_score(self, logits, lens):
        ''' Caculate all path score
            Para:
                logits: state score vec (batch_size, seq_lens, n_labels)
                lens: (batch_size,)
            Return: all path score score 
        '''
        batch_size = logits.shape[0]

        previous = logits.new_full((batch_size, self.n_labels), -inf)
        previous[:, self.start] = 0
        
        for logit in logits.transpose(1, 0):

            previous_exp = previous.unsqueeze(-1).expand(batch_size, self.n_labels, self.n_labels)
            trans_exp = self.transition.unsqueeze(0).expand_as(previous_exp)
            logit_exp = logit.unsqueeze(1).expand(batch_size, self.n_labels, self.n_labels)
            score_mat = previous_exp + trans_exp + logit_exp  
            previous_nxt = self.log_sum_exp(score_mat, dim=1).squeeze(1)
            mask = (lens > 0).float().unsqueeze(-1).expand_as(previous)
            previous = mask * previous_nxt + (1 - mask) * previous
            lens = lens - 1

        score_mat = previous + self.transition[:, self.end].unsqueeze(0).expand_as(previous)
        score = self.log_sum_exp(score_mat, dim=1)

        return score

    def get_loss(self, logits, labels, lens):
        ''' Negative log likelihood loss 
            Para:
                logits: state score vec (batch_size, seq_lens, n_labels)
                labels: (batch_size, seq_lens)
                lens: (batch_size,)
            Return: LOSS
        '''
        gold_score = self.get_gold_score(logits, labels, lens)
        all_score  = self.get_all_score(logits, lens)
        return all_score - gold_score


    def viterbi_decode(self, logits, lens):
        ''' Viterbi Algorithm
            Para:
                logits: (batch_size, seq_len, n_labels)
                lens: (batch_size,)
            Return: 
                score: score vec
                paths: possible path for batch
        '''
        batch_size = logits.shape[0]

        vit = logits.new_full((batch_size, self.n_labels), -inf)
        vit[:, self.start] = 0

        pointers = []
        for logit in logits.transpose(1, 0):

            vit_exp = vit.unsqueeze(-1).expand(batch_size, self.n_labels, self.n_labels)
            trans_exp = self.transition.unsqueeze(0).expand_as(vit_exp)

            vit_trans_sum = vit_exp + trans_exp
            vit_max, vit_argmax = torch.max(vit_trans_sum, dim=1)

            vit_nxt = vit_max + logit
            pointers.append(vit_argmax)

            mask = (lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[:, self.end].unsqueeze(0).expand_as(vit_nxt)

            lens = lens - 1

        pointers = torch.stack(pointers, dim=0)
        scores, idx = vit.max(dim=1)
        paths = [idx]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, dim=1, index=idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx)
        
        scores = scores.squeeze(-1)
        paths = torch.stack(paths[1:], dim=1)
        
        return scores, paths


if __name__ == '__main__':
    
    #1. test for state score
    lens = torch.LongTensor([3, 2])
    labels = torch.LongTensor([[0, 1, 1], 
                               [2, 2, 2]])
    logits = torch.FloatTensor([[[0.4, 0.2, 0.2, 0.1, 0.1], 
                                 [0.2, 0.4, 0.2, 0.1, 0.1], 
                                 [0.2, 0.4, 0.2, 0.1, 0.1]], 

                                [[0.1, 0.1, 0.6, 0.1, 0.1], 
                                 [0.1, 0.1, 0.6, 0.1, 0.1], 
                                 [0.1, 0.1, 0.6, 0.1, 0.1]]])
    crf = CRF()
    new_logits = crf.pad_logits(logits)
    state_score = crf.get_state_score(new_logits, labels, lens)
    #2. test for trans score
    trans_score = crf.get_trans_score(labels, lens)
    #3. test for real path
    real_score = crf.get_gold_score(new_logits, labels, lens)
    #4. test for all path
    all_score = crf.get_all_score(new_logits, lens)
    #5. test for viterbi   
    score, path = crf.viterbi_decode(new_logits, lens)
    print(all_score, real_score, state_score, trans_score)
    #path: [0, 1, 1]
    #      [2, 2, 2]