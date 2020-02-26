import torch
from pybert.configs.basic_config import config
from pybert.io.bert_processor import BertProcessor
from pybert.model.bert_for_multi_label import BertForMultiLable

def main(text,arch,max_seq_length,do_lower_case):
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'] /f'{arch}', num_labels=len(label_list))
    tokens = processor.tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1, 2 choices
    logits = model(input_ids)
    probs = logits.sigmoid()
    return probs.cpu().detach().numpy()[0]

if __name__ == "__main__":
    text = ''''"FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!"'''
    max_seq_length = 256
    do_loer_case = True
    arch = 'bert'
    probs = main(text,arch,max_seq_length,do_loer_case)
    print(probs)
    
  '''
  #output
  [0.98304486 0.40958735 0.9851305  0.04566246 0.8630512  0.07316463]
  '''
