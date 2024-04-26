from transformers import BertTokenizer
import json
import unicodedata

def get_tokenizer():
    """添加特殊中文字符和未使用的token【unused1】"""
    added_token=['[unused1]','[unused1]']
    tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/Desktop/pytorch_bert_bilstm_crf_ner-main/model_hub/chinese-bert-wwm-ext',additional_special_tokens=added_token)
    special_tokens_dict = {'additional_special_tokens':['”','“']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

tokenizer=get_tokenizer()

class token_rematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
            """获取token的“词干”（如果是##开头，则自动去掉##）
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """控制类字符判断
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """判断是不是有特殊含义的符号
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

"""
def load_schema(file_path=args.schema_path):
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        event_type_dict=dict()
        for line in lines:
            data=json.loads(line)
            event_type = data['event_type']
            role_list=[]
            for role in data['role_list']:
                role_list.append(role['role'])
            event_type_dict[event_type]=role_list
        return event_type_dict
"""

def load_schema():
    # 读取schema
    
    label2id={'nr':0,'ns':1,'nt':2}
    id2label={0:'nr',1:'ns',2:'nt'}
    num_labels=3

    return label2id,id2label,num_labels

def get_labels_num():
    _,_,num_labels = load_schema()
    return 2*num_labels+1


def load_vocab():
    with open('./data/vocab.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id,len(word2id)


if __name__=='__main__':

    schema=load_schema()
    print(len(schema))
    # cl=token_rematch()
    # d=cl.rematch(s,c)
    # print(d)

