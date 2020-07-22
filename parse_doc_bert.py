# encoding=utf8
import os
import jieba
import re
import pickle
import numpy as np
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import SentenceSplitter
from scipy.spatial.distance import cosine
from bert_serving.client import BertClient


cws_model_path = os.path.join(os.path.abspath('./'), 'ltp_Model', 'cws.model')
pos_model_path = os.path.join(os.path.abspath('./'), 'ltp_Model', 'pos.model')
par_model_path = os.path.join(
    os.path.abspath('./'), 'ltp_Model', 'parser.model')
ner_model_path = os.path.join(os.path.abspath('./'), 'ltp_Model', 'ner.model')

say_words_path = os.path.join(os.path.abspath(
    './'), 'data', 'saying_words.pickle')
segmentor = Segmentor()  # 分词
postagger = Postagger()  # 词性标注
recognizer = NamedEntityRecognizer()  # 命名主体识别
parser = Parser()  # 依存分析
segmentor.load(cws_model_path)
postagger.load(pos_model_path)
recognizer.load(ner_model_path)
parser.load(par_model_path)


# load saying words
say_words = pickle.load(open(say_words_path, 'rb'))

# 句子依存分析


def parsing(sentence):
    words = segmentor.segment(sentence)  # pyltp分词
    postags = postagger.postag(words)  # 词性标注
    arcs = parser.parse(words, postags)  # 句法分析
    return arcs

# 命名实体识别


def get_name_entity(sentence):
    # sentence = ''.join(strs)
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)  # 词性标注
    netags = recognizer.recognize(words, postags)  # 命名实体识别
    return netags


def get_pos(sentence):
    # sentence = ''.join(strs)
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)  # 词性标注
    return postags
# 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语 find complete subject


# def get_name(name, predic, words, property, ne):
#     index = words.index(name)
#     cut_property = property[index + 1:]  # 截取到name后第一个词语
#     pre = words[:index]  # 前半部分
#     pos = words[index+1:]  # 后半部分
#     # 向前拼接主语的定语
#     l = len(pre)-1
#     for index, values in enumerate(pre):
#         w_index = l - index
#         w = pre[w_index]
#         if property[w_index] == 'ADV':
#             continue
#         if property[w_index] in ['ATT', 'SBV']:  # SBV?
#             name = w + name
#         else:
#             break
#     return name

# 获取谓语之后的言论


def get_saying(sentence, wp, heads, pos):
    # word = sentence.pop(0) #谓语
    # if '：' in sentence:
    #     #         print('hear I am')
    #     return ''.join(sentence[sentence.index('：') + 1:])
    proper = [w.relation for w in wp]
    while pos < len(sentence):
        w = sentence[pos]
        p = proper[pos]
        h = heads[pos]
        #         print('tttt:',w)
        # 谓语尚未结束

        if p in ['DBL', 'CMP', 'RAD']:
            pos += 1
            continue
        # 定语
        # if p == 'ATT' and proper[h - 1] != 'SBV':
        #     pos = h
        #     continue
        # 宾语:直接宾语，间接宾语
        if p in ['VOB', 'IOB']:
            pos += 1
            continue
        # 处理报道<称>，提醒<说>这种情况
        if w == '说' or w == '称':
            pos += 1
            continue
        else:
            if w == '，' or w == '：':  # 说后面逗号跟着言论
                return ''.join(sentence[pos + 1:])
            else:  # 说后面直接跟着言论
                return ''.join(sentence[pos:])

# parse sentense
# boolean allnoun 控制是否把说前面的所有名字都作为name提取出来
def get_name(k, v, cuts,ne,pos,allnoun ):
    name=''
    saying=''
    names=[]
    if ne[k].__contains__('-N'):
        if ne[k] in ['S-Ns', 'S-Ni', 'S-Nh']:
            name = cuts[k]
        # 如果主语是实体的第一部分，则从左到右搜索
        elif ne[k].startswith('B-N'):
            names = [cuts[k]]
            for w, n in zip(cuts[k+1:], ne[k+1:]):
                if n.startswith('I-N'):
                    names.append(w)
                elif n.startswith('E-N'):
                    names.append(n)
                    break
        # 如果主语是实体的最后一部分，则反响搜索
        elif ne[k].startswith('E-N'):
            names = [cuts[k]]
            l = len(ne[:k])-1
            for index, _ in enumerate(zip(cuts[:k], ne[:k])):
                if ne[l-index].startswith('I-N'):
                    names.insert(0,cuts[l-index])
                elif ne[l-index].startswith('B-N'):
                    names.insert(0,cuts[l-index])
                    break
            name = ''.join(names)        
    name = ''.join(re.findall('\w+', name))
    if allnoun and (not name) and pos[k] in ['n','nh','ni','nl','ns','nz'] :
         name = cuts[k] 
    return name

def parse_sentence(sentence,  ws=False):
    cuts = list(segmentor.segment(sentence))  # pyltp分词
    if not cuts:
        return False
    pos = get_pos(sentence)
    ne = list(get_name_entity(sentence))
    wp = parsing(sentence)  # 依存分析
    for k, v in enumerate(wp):
        # 确定第一个主谓句,notice that parsed word_relation list index starts from 1
        if v.relation == 'SBV' and (cuts[v.head-1] in say_words):
            # get name entity
            name = get_name(k, v, cuts,ne,pos,False)
            if name : 
                saying = get_saying(cuts, wp, [
                                    i.head for i in wp], v.head)
                return name, saying
    # name_saying_pairs = re.findall('(\w+)?：([^。！？；]+)',sentence)
    # if name_saying_pairs: return name_saying_pairs[0][0],name_saying_pairs[0][1]
    return False

# split doc into sentences
def split_sentence(doc):
    # 提取类似“不要怕”黎明说，“天会亮”，并把前后的言论都合并到后面
    # 黎明说，“不要怕，天会亮”
    for_back_sens = re.findall(r'“(?:[^“]+)”(?:[^“\r\n]+)，“(?:[^“]+)”',doc)
    try:
        for_back_sens = list(map(lambda x : re.findall(r'”(.*?)“',x)[0] + '“'+ 
    ''.join(re.findall(r'“(.*?)”',x)) + '”',for_back_sens))
    except IndexError :
        print(r'index error')
    
    # 从原文中移除前后皆有的言论
    doc = re.sub(r'“(?:[^“]+)”(?:[^“\r\n]+)，“(?:[^“]+)”','',doc)
    # replace conditions like  。” with mark <period>”。
    doc = re.sub('(。|!|！|？|\.|\?)”', '<period>”。', doc)
    doc = re.sub('"', '', doc)
    # handle differet kinds of \r\n
    doc = re.sub('(\n\r)|(\n)|(\r)', '\r\n<para_start>', doc)
    # split doc into sentences
    sens = re.split('[。!！\.？\?\r\n]', doc)
    sens = list(filter(lambda x: len(x) > 3, sens))
    sens = list(filter(str.strip, sens))
    sens = list(filter(lambda x: not (
        (x.startswith('<para_start>') and len(x) < 15)), sens))

    # merge sentences between quotes “”
    new = []
    for index, s in enumerate(sens):
        if ('“' in s) and (s.count('“')-s.count('”') == 1):
            for index1, s1 in enumerate(sens[index + 1:]):
                if s1.startswith('<para_start>'):
                    break
                if ('”' in s1) and (s1.count('”')-s1.count('“') == 1):
                    new.append(','.join(sens[index: index + 1 + index1 + 1]))
                    sens[index: index + 1 + index1 +
                         1] = ['processed' for i in range(1 + index1 + 1)]
                    break
        else:
            if s != 'processed':
                new.append(s)
    # remove sentense with only  '<para_start>' as content
    new1 = []
    while new:
        s = new.pop(0)
        if ((len(str.strip(s)) > 0) and s != '<para_start>'):
            new1.append(s)

    # add periods after last right quote for better entity recognition
    doc_perods = ('<perods>'.join(new1)).replace('<period>”', '。”')
    new1 = doc_perods.split('<perods>')
    return new1 + for_back_sens

def clean_name_beta(name):
    cuts = list(jieba.cut(name))
    name = ''.join([cuts[index] for index, a in enumerate(
        list(get_name_entity(name))) if 'N' in list(a)])
    return name

def sents_similar(sen1, sen2,bc):
    sens_embedings = bc.encode([sen1,sen2])
    return cosine(sens_embedings[0],sens_embedings[1])

# parse documents, which will be spiltted into sentenses
def parse_doc(threshhold, doc,bc):
    result = []
    sens = split_sentence(doc)
    original_sens = sens.copy()
    original_index = 0
    while sens:
        first_sen = sens.pop(0)
        first_sen = first_sen.replace('<para_start>', '')
        saying = parse_sentence(first_sen, ws=False)
        if saying and saying[1]:
            saying_index = original_index
            extended_saying = saying[1]
            # check the followding setence, if they have the same meaning as the previouse one, then concatenate as the same saying
            temp_sens = [first_sen] + sens  # add first_sens for easy comparing
            for index, pre in enumerate(temp_sens):
                if index == len(temp_sens)-1:
                    break
                if (not temp_sens[index + 1].startswith('<para_start>')) and \
                        sents_similar(pre,temp_sens[index + 1],bc) < threshhold:
                    extended_saying = extended_saying + \
                        '，' + temp_sens[index + 1]
                    sens.pop(0)
                    original_index += 1
                else:
                    break
            # saying[0] indicates the one gives comment
            result.append((saying[0], extended_saying, saying_index))
        original_index += 1
    return result


def test(doc,bc):
    result = parse_doc(0.15, doc,bc)
    for index, s in enumerate(result):
        print(index, s, '\n')

if __name__ == '__main__':
    bc = BertClient()
    doc = '''据南方周末报道，“如果我们对这个人没有其他了解，我们知道他的优先事项完全是个人的、自恋的。我们知道，他并不担心国家的稳定或安全，鉴于这一系列心理现实，我们需要一个比我们所能保证的任何程度的信心更加坚定的进程，才能确保我们将在明年1月顺利、和平地过渡到一位新总统。”'''
    test(doc,bc)
    # print(list(get_pos(doc)),'\n',list(segmentor.segment(doc)))
