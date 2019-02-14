#encoding:utf-8
import csv
import numpy as np
from torch.utils.data import Dataset

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid   = guid  # 该样本的唯一ID
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    数据的feature集合
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids   = input_ids   # tokens的索引
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id

class CreateDataset(Dataset):
    def __init__(self,data_path,max_seq_len,tokenizer,example_type,seed):
        self.seed    = seed
        self.max_seq_len  = max_seq_len
        self.example_type = example_type
        self.data_path  = data_path
        self.tokenizer = tokenizer
        self.reset()

    # 初始化
    def reset(self):
        # 构建examples
        self.build_examples()

    # 读取数据集
    def read_data(self,quotechar = None):
        '''
        默认是以tab分割的数据
        :param quotechar:
        :return:
        '''
        lines = []
        with open(self.data_path,'r',encoding='utf-8') as fr:
            reader = csv.reader(fr,delimiter = '\t',quotechar = quotechar)
            for line in reader:
                lines.append(line)
        return lines

    # 构建数据examples
    def build_examples(self):
        lines = self.read_data()
        self.examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(self.example_type,i)
            label = [np.float32(x) for x in line[0].split(',')] # RuntimeError: expected type torch.cuda.DoubleTensor but got torch.cuda.FloatTensor
            text_a = line[1]
            example = InputExample(guid = guid,text_a = text_a,label= label)
            self.examples.append(example)
        del lines

    # 将example转化为feature
    def build_features(self,example):
        '''
        # 对于两个句子:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # 对于单个句子:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # type_ids:表示是第一个句子还是第二个句子
        '''
        #转化为token
        tokens_a = self.tokenizer.tokenize(example.text_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:(self.max_seq_len - 2)]
        # 句子首尾加入标示符
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)  # 对应type_ids
        # 将词转化为语料库中对应的id
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 输入mask
        input_mask = [1] * len(input_ids)
        # padding，使用0进行填充
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        # 标签
        label_id = example.label
        feature = InputFeature(input_ids = input_ids,input_mask = input_mask,
                               segment_ids = segment_ids,label_id = label_id)
        return feature

    def _preprocess(self,index):
        example = self.examples[index]
        feature = self.build_features(example)
        return np.array(feature.input_ids),np.array(feature.input_mask),\
               np.array(feature.segment_ids),np.array(feature.label_id)

    def __getitem__(self, index):
        return self._preprocess(index)

    def __len__(self):
        return len(self.examples)
