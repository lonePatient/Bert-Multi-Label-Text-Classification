import torch
def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_input_mask = all_input_mask[:, -max_len:]
    all_segment_ids = all_segment_ids[:, -max_len:]
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

