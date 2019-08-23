from collections import Counter
from ..common.tools import save_pickle
from ..common.tools import load_pickle
from ..common.tools import logger

class Vocabulary(object):
    def __init__(self, max_size=None,
                 min_freq=None,
                 pad_token="[PAD]",
                 unk_token = "[UNK]",
                 cls_token = "[CLS]",
                 sep_token = "[SEP]",
                 mask_token = "[MASK]",
                 add_unused = False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.word2id = {}
        self.id2word = None
        self.rebuild = True
        self.add_unused = add_unused
        self.word_counter = Counter()
        self.reset()

    def reset(self):
        ctrl_symbols = [self.pad_token,self.unk_token,self.cls_token,self.sep_token,self.mask_token]
        for index,syb in enumerate(ctrl_symbols):
            self.word2id[syb] = index

        if self.add_unused:
            for i in range(20):
                self.word2id[f'[UNUSED{i}]'] = len(self.word2id)

    def update(self, word_list):
        '''
        依次增加序列中词在词典中的出现频率
        :param word_list:
        :return:
        '''
        self.word_counter.update(word_list)

    def add(self, word):
        '''
        增加一个新词在词典中的出现频率
        :param word:
        :return:
        '''
        self.word_counter[word] += 1

    def has_word(self, word):
        '''
        检查词是否被记录
        :param word:
        :return:
        '''
        return word in self.word2id

    def to_index(self, word):
        '''
        将词转为数字. 若词不再词典中被记录, 将视为 unknown, 若 ``unknown=None`` , 将抛出
        :param word:
        :return:
        '''
        if word in self.word2id:
            return self.word2id[word]
        if self.unk_token is not None:
            return self.word2id[self.unk_token]
        else:
            raise ValueError("word {} not in vocabulary".format(word))

    def unknown_idx(self):
        """
        unknown 对应的数字.
        """
        if self.unk_token is None:
            return None
        return self.word2id[self.unk_token]

    def padding_idx(self):
        """
        padding 对应的数字
        """
        if self.pad_token is None:
            return None
        return self.word2id[self.pad_token]

    def to_word(self, idx):
        """
        给定一个数字, 将其转为对应的词.

        :param int idx: the index
        :return str word: the word
        """
        return self.id2word[idx]

    def build_vocab(self):
        max_size = min(self.max_size, len(self.word_counter)) if self.max_size else None
        words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self.word2id:
            words = filter(lambda kv: kv[0] not in self.word2id, words)
        start_idx = len(self.word2id)
        self.word2id.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        logger.info(f"The size of vocab is: {len(self.word2id)}")
        self.build_reverse_vocab()
        self.rebuild = False

    def save(self, file_path):
        '''
        保存vocab
        :param file_name:
        :param pickle_path:
        :return:
        '''
        mappings = {
            "word2id": self.word2id,
            'id2word': self.id2word
        }
        save_pickle(data=mappings, file_path=file_path)

    def save_bert_vocab(self,file_path):
        bert_vocab = [x for x,y in self.word2id.items()]
        with open(str(file_path),'w') as fo:
            for token in bert_vocab:
                fo.write(token+"\n")

    def load_from_file(self, file_path):
        '''
        从文件组红加载vocab
        :param file_name:
        :param pickle_path:
        :return:
        '''
        mappings = load_pickle(input_file=file_path)
        self.id2word = mappings['id2word']
        self.word2id = mappings['word2id']

    def build_reverse_vocab(self):
        self.id2word = {i: w for w, i in self.word2id.items()}

    def clear(self):
        """
        删除Vocabulary中的词表数据。相当于重新初始化一下。
        :return:
        """
        self.word_counter.clear()
        self.word2id = None
        self.id2word = None
        self.rebuild = True
        self.reset()

    def __len__(self):
        return len(self.id2word)
