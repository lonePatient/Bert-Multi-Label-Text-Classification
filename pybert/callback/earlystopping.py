import numpy as np
from ..common.tools import logger
class EarlyStopping(object):
    '''
        """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.

    # Arguments
        min_delta: 最小变化
        patience: 多少个epoch未提高，就停止训练
        verbose: 信息大于，默认打印信息
        mode: 计算模式
        monitor: 计算指标
        baseline: 基线
    '''
    def __init__(self,
                 min_delta = 0,
                 patience  = 10,
                 verbose   = 1,
                 mode      = 'min',
                 monitor   = 'loss',
                 baseline  = None):

        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor

        assert mode in ['min','max']

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.reset()

    def reset(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stop_training = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_step(self,current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose >0:
                    logger.info(f"{self.patience} epochs with no improvement after which training will be stopped")
                self.stop_training = True
