#encoding:utf-8
import time
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import restore_checkpoint,model_device
from ..utils.utils import summary
# 训练包装器
class Trainer(object):
    def __init__(self,train_configs):

        self.start_epoch = 1
        self.global_step = 0
        self.n_gpu = train_configs['n_gpu']
        self.model = train_configs['model']
        self.epochs = train_configs['epochs']
        self.logger = train_configs['logger']
        self.verbose = train_configs['verbose']
        self.criterion = train_configs['criterion']
        self.optimizer = train_configs['optimizer']
        self.lr_scheduler = train_configs['lr_scheduler']
        self.early_stopping = train_configs['early_stopping']
        self.epoch_metrics = train_configs['epoch_metrics']
        self.batch_metrics = train_configs['batch_metrics']
        self.model_checkpoint = train_configs['model_checkpoint']
        self.training_monitor = train_configs['training_monitor']
        self.gradient_accumulation_steps = train_configs['gradient_accumulation_steps']

        self.model, self.device = model_device(n_gpu = self.n_gpu, model=self.model, logger=self.logger)
        # 重载模型，进行训练
        if train_configs['resume']:
            self.logger.info(f"\nLoading checkpoint: {train_configs['resume']}")
            resume_list = restore_checkpoint(resume_path =train_configs['resume'],model = self.model,optimizer = self.optimizer)
            best = resume_list[2]
            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            self.start_epoch = resume_list[3]
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{train_configs['resume']}' and epoch {self.start_epoch} loaded")

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


    def _save_info(self,epoch,valid_loss):
        '''
        保存模型信息
        '''
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'valid_loss': round(valid_loss,4)
        }
        return state

    def _valid_epoch(self,data):
        '''
        valid数据集评估
        '''
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(data):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                label = label_ids.to(self.device)
                logits = self.model(input_ids, segment_ids,input_mask)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(label.cpu().detach())

            self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
            self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
            loss = self.criterion(target = self.targets, output=self.outputs)
            self.result['valid_loss'] = loss.item()
            print("\n--------------------------valid result ------------------------------")
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'valid_{metric.name()}'] = value
            if len(self.n_gpu) > 0:
                torch.cuda.empty_cache()
            return self.result

    def _train_epoch(self,data):
        '''
        epoch训练
        :param data:
        :return:
        '''
        self.epoch_reset()
        self.model.train()
        for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(data):
            start = time.time()
            self.batch_reset()
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label = label_ids.to(self.device)
            logits = self.model(input_ids, segment_ids,input_mask)
            # 计算batch loss
            loss = self.criterion(output=logits,target=label)
            if len(self.n_gpu) >= 2:
                loss = loss.mean()
            # 如果梯度更新累加step>1，则也需要进行mean操作
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward()
            # 学习率更新方式
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits = logits,target = label)
                    self.info[metric.name()] = metric.value()

            self.info['loss'] = loss.item()
            if self.verbose >= 1:
                self.progressbar.batch_step(batch_idx= step,info = self.info,use_time=time.time() - start)
            # 为了降低显存使用量
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label.cpu().detach())

        print("\n------------------------- train result ------------------------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim =0).cpu().detach()
        self.targets = torch.cat(self.targets, dim =0).cpu().detach()
        loss = self.criterion(target=self.targets, output=self.outputs)
        self.result['loss'] = loss.item()

        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if len(self.n_gpu) > 0:
            torch.cuda.empty_cache()
        return self.result

    def train(self,train_data,valid_data):
        self.batch_num = len(train_data)
        self.progressbar = ProgressBar(n_batch=self.batch_num)

        print("model summary info: ")
        for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_data):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            summary(self.model,*(input_ids, segment_ids,input_mask),show_input=True)
            break
        # ***************************************************************
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print("--------------------Epoch {epoch}/{self.epochs}------------------------")
            train_log = self._train_epoch(train_data)
            valid_log = self._valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key,value in logs.items()])
            self.logger.info(show_info)
            print("-----------------------------------------------------------------------")
            # 保存训练过程中模型指标变化
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self._save_info(epoch,valid_loss = logs['valid_loss'])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor],state = state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break


