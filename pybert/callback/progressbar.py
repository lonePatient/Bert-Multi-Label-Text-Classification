#encoding:Utf-8
class ProgressBar(object):
    def __init__(self,n_batch,
                 width=30):
        self.width = width
        self.n_batch = n_batch
    def batch_step(self,batch_idx,info,use_time):
        recv_per = int(100 * (batch_idx + 1) / self.n_batch)
        if recv_per >= 100:
            recv_per = 100
        # 进度条模式
        show_bar = f"\r[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        # 打印信息
        show_info = f'\r[training] {batch_idx+1}/{self.n_batch} {show_bar} -{use_time:.1f}s/step '+\
                   "-".join([f' {key}: {value:.4f} ' for key,value in info.items()])
        print(show_info,end='')


