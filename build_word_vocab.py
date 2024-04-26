import json
import jieba
# from utils_new.arguments_parse import args
import tools
from tqdm import tqdm

def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        lines = json.load(f)
        for l in lines:
            D.append(l['text'])
    return D


def save_vocab():
    stop_words = set()
    with open('/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")   # 去掉读取每一行数据的\n
            stop_words.add(line)
    word_list=[]
    sentences = load_data('/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/cner/mid_data/train.json')
    sentences.extend(load_data('/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/cner/mid_data/dev.json'))
    sentences.extend(load_data('/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/data/cner/mid_data/test.json'))
    for sent in tqdm(sentences):
        tmp_word_list=list(jieba.cut(sent,cut_all=True))
        for word in tmp_word_list:
            if word not in word_list and word not in stop_words:
                word_list.append(word)
                
    vocab_lenth=len(word_list)
    word2id={}
    id2word={}
    for i,word in enumerate(word_list):
        word2id[word]=i+1
        id2word[i+1]=word

    with open('./data/vocab.json','w',encoding='utf8') as f:
        tmp=json.dumps(word2id,ensure_ascii=False)
        f.write(tmp)

    return word2id,id2word,vocab_lenth


def load_vocab():
    with open('./data/vocab.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id,len(word2id)

if __name__=="__main__":
    save_vocab()
    # word,l=load_vocab()
