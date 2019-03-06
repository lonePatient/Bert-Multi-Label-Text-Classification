#encoding:utf-8
import torch
import numpy as np
from ..utils.utils import model_device,load_bert

class Predicter(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu,
                 model_path
                 ):
        self.model = model
        self.logger = logger
        self.width = 30
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model, logger=self.logger)
        loads = load_bert(model_path=model_path,model = self.model)
        self.model = loads[0]

    def show_info(self,batch_id,n_batch):
        recv_per = int(100 * (batch_id + 1) / n_batch)
        if recv_per >= 100:
            recv_per = 100
        # 进度条模式
        show_bar = f"\r[predict]{batch_id+1}/{n_batch}[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        print(show_bar,end='')

    def predict(self,data):
        all_logits = None
        self.model.eval()
        n_batch = len(data)
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(data):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
                self.show_info(step,n_batch)
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
        return all_logits






