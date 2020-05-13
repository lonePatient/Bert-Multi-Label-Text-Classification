import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import summary
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self,args,model,logger,criterion,optimizer,scheduler,early_stopping,epoch_metrics,
                 batch_metrics,verbose = 1,training_monitor = None,model_checkpoint = None
                 ):
        self.args = args
        self.model = model
        self.logger =logger
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu = args.n_gpu, model=self.model)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.resume_path:
            self.logger.info(f"\nLoading checkpoint: {args.resume_path}")
            resume_dict = torch.load(args.resume_path / 'checkpoint_info.bin')
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{args.resume_path}' and epoch {self.start_epoch} loaded")

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self,epoch,best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model":model_save,
                 'epoch':epoch,
                 'best':best}
        return state

    def valid_epoch(self,data):
        pbar = ProgressBar(n_total=len(data),desc="Evaluating")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids,input_mask)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        loss = self.criterion(target = self.targets, output=self.outputs)
        self.result['valid_loss'] = loss.item()
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train_epoch(self,data):
        pbar = ProgressBar(n_total = len(data),desc='Training')
        tr_loss = AverageMeter()
        self.epoch_reset()
        for step,  batch in enumerate(data):
            self.batch_reset()
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, segment_ids,input_mask)
            loss = self.criterion(output=logits,target=label_ids)
            if len(self.args.n_gpu) >= 2:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits = logits,target = label_ids)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            tr_loss.update(loss.item(),n = 1)
            if self.verbose >= 1:
                pbar(step= step,info = self.info)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        print("\n------------- train result --------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim =0).cpu().detach()
        self.targets = torch.cat(self.targets, dim =0).cpu().detach()
        self.result['loss'] = tr_loss.avg
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self,train_data,valid_data):
#         print("model summary info: ")
#         for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_data):
#             input_ids = input_ids.to(self.device)
#             input_mask = input_mask.to(self.device)
#             segment_ids = segment_ids.to(self.device)
#             summary(self.model,*(input_ids, segment_ids,input_mask),show_input=True)
#             break
        # ***************************************************************
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch,self.start_epoch+self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key,value in logs.items()])
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs) 

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch,best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor],state = state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break






