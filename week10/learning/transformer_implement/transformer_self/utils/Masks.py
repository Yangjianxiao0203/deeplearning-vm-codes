import torch


def get_pad_mask(seq, pad_idx,n_head):
    '''
    For masking out the padding part of sequence.
    seq: [batch_size, seq_len]

    return: batch_size * n_head * seq_len * seq_len
    '''
    # Create initial mask
    attention_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # shape: [batch_size, 1, 1, seq_len]

    # Expand to shape [batch_size, 1, seq_len, seq_len]
    attention_mask_expanded = attention_mask.expand(-1, 1, seq.size(1), -1)  # Expand the 3rd dimension

    # Multiply with transpose to get the final mask
    mask = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)

    # Repeat for all heads
    mask = mask.repeat(1, n_head, 1, 1)

    return mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask