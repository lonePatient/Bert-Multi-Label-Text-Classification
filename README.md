## Bert multi-label text classification by PyTorch

This repo contains a PyTorch implementation of the pretrained BERT and XLNET model for multi-label text classification.

###  Structure of the code

At the root of the project, you will see:

```text
├── pybert
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── pretrain　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── common # a set of utility functions
├── run_bert.py
├── run_xlnet.py
```
### Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- pandas
- pytorch_transformers=1.0.0

**note**: pytorch_transformers>=1.2.0, modify `self.apply(self.init_weights)` to `self.init_weights()` in `pybert/model/nn/bert_for_multi_label.py file` .

### How to use the code

you need download pretrained bert model and xlnet model.

<div class="note info"><p> BERT:  bert-base-uncased</p></div>
<div class="note info"><p> XLNET:  xlnet-base-cased</p></div>

1. Download the Bert pretrained model from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 
2. Download the Bert config file from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json) 
3. Download the Bert vocab file from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) 
4. Rename:

    - `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin`
    - `bert-base-uncased-config.json` to `config.json`
    - `bert-base-uncased-vocab.txt` to `bert_vocab.txt`
5. Place `model` ,`config` and `vocab` file into  the `/pybert/pretrain/bert/base-uncased` directory.
6. `pip install pytorch-transformers` from [github](https://github.com/huggingface/pytorch-transformers).
7. Download [kaggle data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place in `pybert/dataset`.
    -  you can modify the `io.task_data.py` to adapt your data.
8. Modify configuration information in `pybert/configs/basic_config.py`(the path of data,...).
9. Run `python run_bert.py --do_data` to preprocess data.
10. Run `python run_bert.py --do_train --save_best --do_lower_case` to fine tuning bert model.
11. Run `run_bert.py --do_test --do_lower_case` to predict new data.

### training 

```text
[training] 8511/8511 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] -0.8s/step- loss: 0.0640
training result:
[2019-01-14 04:01:05]: bert-multi-label trainer.py[line:176] INFO  
Epoch: 2 - loss: 0.0338 - val_loss: 0.0373 - val_auc: 0.9922
```
### training figure

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190214210111.png)

### result

```python
---- train report every label -----
Label: toxic - auc: 0.9903
Label: severe_toxic - auc: 0.9913
Label: obscene - auc: 0.9951
Label: threat - auc: 0.9898
Label: insult - auc: 0.9911
Label: identity_hate - auc: 0.9910
---- valid report every label -----
Label: toxic - auc: 0.9892
Label: severe_toxic - auc: 0.9911
Label: obscene - auc: 0.9945
Label: threat - auc: 0.9955
Label: insult - auc: 0.9903
Label: identity_hate - auc: 0.9927
```

## Tips

- When converting the tensorflow checkpoint into the pytorch, it's expected to choice the "bert_model.ckpt", instead of "bert_model.ckpt.index", as the input file. Otherwise, you will see that the model can learn nothing and give almost same random outputs for any inputs. This means, in fact, you have not loaded the true ckpt for your model
- When using multiple GPUs, the non-tensor calculations, such as accuracy and f1_score, are not supported by DataParallel instance
- As recommanded by Jocob in his paper <url>https://arxiv.org/pdf/1810.04805.pdf<url/>, in fine-tuning tasks, the hyperparameters are expected to set as following: **Batch_size**: 16 or 32, **learning_rate**: 5e-5 or 2e-5 or 3e-5, **num_train_epoch**: 3 or 4
- The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256 
- Upon testing, we found that fine-tuning all layers could get much better results than those of only fine-tuning the last classfier layer. The latter is actually a feature-based way 
