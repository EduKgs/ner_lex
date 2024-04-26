# pytorch_bert_bilstm_crf_ner

# 依赖
```python
python==3.6 (可选)
pytorch==1.6.0 (可选)
pytorch-crf==0.7.2
transformers==4.5.0
numpy==1.22.4
packaging==21.3
```

****
这里总结下步骤，以cner数据为例：
```python
先去hugging face下载相关文件到chinese-bert-wwwm-ext下。
目录结构：
--pytorch_bilstm_crf_ner
--model_hub
----chinese-bert-wwm-ext
------vocab.txt
------config.json
------pytorch_model.bin

1、原始数据放在data/cner/raw_data/下，并新建mid_data和final_data两个文件夹。
2、将raw_data下的数据处理成mid_data下的格式。其中：
--labels.txt：实体类别
["PRO", "ORG", "CONT", "RACE", "NAME", "EDU", "LOC", "TITLE"]
--nor_ent2id.json：BIOES格式的标签
{"O": 0, "B-PRO": 1, "I-PRO": 2, "E-PRO": 3, "S-PRO": 4, "B-ORG": 5, "I-ORG": 6, "E-ORG": 7, "S-ORG": 8, "B-CONT": 9, "I-CONT": 10, "E-CONT": 11, "S-CONT": 12, "B-RACE": 13, "I-RACE": 14, "E-RACE": 15, "S-RACE": 16, "B-NAME": 17, "I-NAME": 18, "E-NAME": 19, "S-NAME": 20, "B-EDU": 21, "I-EDU": 22, "E-EDU": 23, "S-EDU": 24, "B-LOC": 25, "I-LOC": 26, "E-LOC": 27, "S-LOC": 28, "B-TITLE": 29, "I-TITLE": 30, "E-TITLE": 31, "S-TITLE": 32}
--train.json/dev.json/test.json：是一个列表，列表里面每个元素是：
[
  {
    "id": 0,
    "text": "常建良，男，",
    "labels": [
      [
        "T0",
        "NAME",  
        0,
        3,  # 后一位
        "常建良"
      ]
    ]
  },
  ......
]
3、在preprocess.py里面修改数据集名称和设置文本最大长度，并按照其它数据一样添加一段代码。运行后得到final_data下的数据。
4、运行指令进行训练、验证和测试：
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--data_name="cner" \
--model_name="bert" \# 默认为bert
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=33 \# BIOES标签的数目
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \# 文本最大长度，和prepcoess.py里面保持一致
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \# 训练batch_size
--train_epochs=3 \# 训练epoch
--eval_batch_size=32 \# 验证batch_size
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm="False" \# 是否使用bilstm
--use_idcnn="True" \# 是否使用idcnn。idcnn和bilstm只能选择一种
--use_crf="True" \# 是否使用crf
--dropout_prob=0.3 \
--dropout=0.3
```
运行的时候需要在命令行运行，且不要带上后面的注释。windows下运行需要将指令变成一行，即删除掉"\\"。
