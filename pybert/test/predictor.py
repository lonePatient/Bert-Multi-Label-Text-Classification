#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu
                 ):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data))
        all_logits = None
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
                pbar.batch_step(step=step,info = {},bar_type='Testing')
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits






