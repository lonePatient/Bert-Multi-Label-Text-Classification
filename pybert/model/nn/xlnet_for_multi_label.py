import torch.nn as nn
from pytorch_transformers.modeling_xlnet import XLNetPreTrainedModel, XLNetModel,SequenceSummary

class XlnetForMultiLable(XLNetPreTrainedModel):
    def __init__(self, config):

        super(XlnetForMultiLable, self).__init__(config)
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None,attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None,head_mask=None):
        # XLM don't use segment_ids
        token_type_ids = None
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids,
                                    input_mask=input_mask, attention_mask=attention_mask,
                                    mems=mems, perm_mask=perm_mask, target_mapping=target_mapping,
                                    head_mask=head_mask)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.classifier(output)
        return logits
