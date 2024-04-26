# -*- coding:utf-8 -*-
# @Time  : 2020/11/3 14:22
# @Author: yangping

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/cner/mid_data/train.json",help="train file")
parser.add_argument("--test_path", type=str, default="/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/cner/mid_data/dev.json",help="test file")
parser.add_argument("--schema_path", type=str, default="./event_schema/event_schema.json",help="schema")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/multilabel_cls.pth",help="output_dir")
parser.add_argument("--bert_mrc_checkpoints", type=str, default="./checkpoints/bert_mrc.pth",help="output_dir")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=16,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=768,help="hidden_num")
parser.add_argument("--max_length", type=int, default=80,help="max_length")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=80,help="epoch")
parser.add_argument("--learning_rate", type=float, default=1e-5,help="learning_rate")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
parser.add_argument("--pretrained_model_path", type=str, default="./pretrained_model/chinese-bert-wwm-ext",help="pretrained_model_path")
parser.add_argument("--clip_norm", type=str, default=0.25,help="clip_norm")
parser.add_argument("--warm_up_epoch", type=str, default=1,help="warm_up_steps")
parser.add_argument("--decay_epoch", type=str, default=20,help="decay_steps")
parser.add_argument("--weight_decay", type=str, default=0.01,help="weight_decay")
parser.add_argument("--output", type=str, default="./output/result.json",help="output")

args = parser.parse_args()
