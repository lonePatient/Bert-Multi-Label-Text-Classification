#encoding:utf-8
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('agg') # 防止ssh上绘图问题

class TrainingMonitor():
    def __init__(self, file_dir,arch,start_at=0):
        '''
        :param startAt: 重新开始训练的epoch点
        '''
        if isinstance(file_dir,Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.start_at = start_at
        self.H = {}
        self.json_path = file_dir / (arch+"_training_monitor.json")
        self.reset()

    def reset(self):
        if self.start_at > 0:
            # 如果jsonPath文件存在，咋加载历史训练数据
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = json.loads(open(str(self.json_path)).read())
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def epoch_step(self,logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v,np.float):
                v = round(float(v),4)
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            f = open(str(self.json_path), "w")
            f.write(json.dumps(self.H))
            f.close()

        #保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key}') for key in self.H.keys()}
        if len(self.H["loss"]) > 1:
            # 指标变化曲线
            # 需要成对出现
            keys = [key for key,_ in self.H.items() if '_' not in key]
            for key in keys:
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[key],label=f"train_{key}")
                plt.plot(N, self.H[f"valid_{key}"],label=f"valid_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()


