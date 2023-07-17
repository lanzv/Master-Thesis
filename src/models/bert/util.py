import re
from transformers import BertTokenizer

def cut_sentence(text):
    '''
    Cut paragraph into sentences because BERT has a maximum input length of 512.
    '''
    sentences = re.split(r"(？|。|！|……|“|”|‘|’)", text)
    sentences.append("")
    new_sentences = []
    for i in zip(sentences[0::2], sentences[1::2]):
        if ('“' in i) or ('”' in i) or ('‘' in i) or ('’' in i):
            new_sentences.append(i[0])
            new_sentences.append(i[1])
        else:
             new_sentences.append("".join(i))
    new_sentences = [i.strip() for i in new_sentences if len(i.strip()) > 0]

    return new_sentences

def unk_process(text, result_text):
    '''
    Process '[UNK]' because there are some tokens unknown to BERT. 

    Process Uppercase letters, because BERT encode both uppercase letters
    and lowercase letters into the same tokens, which will be decoded into 
    only lowercase letters.

    *** This is not post-process ***
    '''
    # process [UNK]
    if '[UNK]' in result_text:
        origin_tokens = []
        for token in text:
            if (token not in result_text) and not re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', token):
                origin_tokens.append(token)
        for token in origin_tokens:
            result_text = result_text.replace('[UNK]', token, 1)

    # process uppercase and lowercase letters
    letters = []
    for i in text:
        if (re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', i)):
            letters.append(i)
    p = 0
    s1 = list(result_text)
    for i in range(len(result_text)):
        if (re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', s1[i])):
            s1[i] = letters[p]
            p += 1
    result_text = ''.join(s1)
    return result_text

class converter:
    def __init__(self, vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

    def label2text_ids(self, text_ids, labels):
        '''
        Input: a list of token ids and a list of labels,
        Return: a list of words.

        e.g. text_ids=[x1, x2, x3, x4, x5, x6], labels=[0, 0, 1, 0, 0, 1]
            --> [[x1], [x2, x3], [x4], [x5, x6]]
        '''
        #assert len(text_ids) == len(labels)
        word_ids = []
        p = 0
        for i in range(len(labels)):
            if labels[i] == 0: # label 'B'
                word_ids.append(text_ids[p:i])
                p = i
            if i == len(labels) - 1:
                word_ids.append(text_ids[p:])
        return word_ids[1:]

    def text_ids2label(self, word_ids_list):
        '''
        Input: a list of words,
        Return: a list of labels.

        e.g. word_ids_list=[[x1], [x2, x3], [x4], [x5, x6]]
            --> [0, 0, 1, 0, 0, 1]
        '''
        labels = []
        for word in word_ids_list:
            if len(word) == 0:
                continue
            labels.append(0)
            if len(word) == 1:
                continue
            for i in range(len(word) - 1):
                labels.append(1)
        return labels

    def label2text(self, text, labels):
        '''
        Input: a string and a list of labels,
        Return: segmented text.

        e.g. text='我喜欢写代码' (I love writing code) , labels=[0, 0, 1, 0, 0, 1]
            --> '我  喜欢  写  代码'
        '''
        token_ids = []
        i = 0
        while i < len(text):
            token_ids += self.tokenizer.encode(text[i:(i+100)], add_special_tokens=False)
            i += 100
        
        word_ids = self.label2text_ids(token_ids, labels)
        words = [self.tokenizer.decode(ids) for ids in word_ids]
        for i in range(len(words)):
            words[i] = words[i].replace(' ','')
        res = str()
        for word in words:
            res += (word + '  ')
        res = res.replace(' ##', '')
        res = res.replace("  ", "") # last two spaces
        res = unk_process(text, res)
        return res

    def text2label(self, seg_text):
        '''
        Input: segmented text
        Return: a list of labels

        e.g. seg_text='我  喜欢  写  代码' (I love writing code)
            --> [0, 0, 1, 0, 0, 1]
        '''
        words = seg_text.split('  ')
        word_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in words]
        return self.text_ids2label(word_ids)
