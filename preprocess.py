# _*_ coding:utf8 _*_
import json
from tqdm import tqdm
import nltk
import os
import codecs
import sys
import numpy as np
import random
import unicodedata
import spacy
from collections import Counter

spacy_nlp = spacy.load('en_core_web_sm')

reload(sys)
sys.setdefaultencoding('utf8')

random.seed(42)
np.random.seed(42)

SPLIT_TOKEN = "  "


#
# class SingletonDecorator:
#     def __init__(self, klass):
#         self.klass = klass
#         self.instance = None
#
#     def __call__(self, *args, **kwds):
#         if self.instance == None:
#             self.instance = self.klass(*args, **kwds)
#         return self.instance
#
#
# class Spacer:
#     def __init__(self):
#         self.nlp = spacy.load('en_core_web_sm', parser=False)
#
#     def __call__(self, *args, **kwargs):
#         return self.nlp(*args, **kwargs)
#
#
# Spacer = SingletonDecorator(Spacer)


def get_data_from_json(file_name, data_type):
    with open(file_name, 'r') as f:
        data = json.load(f)
        dataset = data['data']
    return dataset


def utf_normalize_text(text):
    return unicodedata.normalize('NFD', text)


def tokenize_pos_ner(sent):
    # tokens = nltk.word_tokenize(sent)
    # nltk.pos_tag(tokens)
    # spacy_nlp = Spacer()

    doc = spacy_nlp(sent)
    tokens = [w.text.replace("``", '"').replace("''", '"').strip() for w in doc]

    # spacy的说明 https://spacy.io/usage/linguistic-features
    # 文章自身的特征（POS、NER、lemma、Term Frequency）
    poses = [w.tag_ for w in doc]  # 获取 POS(pos_: The simple part-of-speech tag; Tag_: The detailed part-of-speech tag.)
    ners = [w.ent_type_ if w.ent_type_ != '' else u"<UNK>" for w in doc]  # 获取token 级别的 NER （命名实体识别）
    lemmas = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in doc]  # 获取词干
    # Term Frequency
    tokens_lower = [w.lower() for w in tokens]
    counter = Counter(tokens_lower)
    total = float(len(tokens_lower))
    tf = [counter[w] / total for w in tokens_lower]

    return tokens, poses, ners, lemmas, tf
    # return [token.replace("``", '"').replace("''", '"').strip() for token in tokens]
    # return [token.replace("``", '"').replace("''", '"').lower() for token in tokens]


def getcharloc2wordloc(context, context_tokens):
    '''
     获取每个 char 对应文本中的第几个 token 的映射表
    :param context: 原始文本
    :param context_tokens:  token 化后的 list
    :return:
    '''
    acc = ''
    word_index = 0  # word 在文本中 的下标
    mapping = {}
    context_token = context_tokens[word_index]
    # if context.startswith('the code itself was patterned so '):
    #     pass
    for i, char in enumerate(context):
        mapping[i] = (context_token, word_index)  # (word,word下标)
        char = char.strip()  # 去除一些奇怪的不可见字符
        if len(char) > 0:
            # if char != u' ' and char != '\n' and char!=u'　' and char!=' ' and char!=' ':
            acc += char
            context_token = context_tokens[word_index]
            if acc == context_token:
                word_index += 1
                acc = ''
    if len(context_tokens) != word_index:
        # print(' '.join(context_tokens))
        # print(context)
        return None
    return mapping


def c2q_match(context_list, question_set):
    '''
    判断文章单词是否出现在问题中
    :param context_list: 文章单词列表
    :param question_set: 问题单词集
    :return:
    '''
    return [(1 if w in question_set else 0) for w in context_list]


def preprocess_and_save_data(data_type, file_name, out_dir):
    '''
    将json数据处理为训练数据，并保存
    :param data_type: train\dev
    :param file_name: 文件名
    :param out_dir: 保存位置
    :return:
    '''

    examples = []
    num_mappingprob, num_tokenprob = 0, 0
    dataset = get_data_from_json(file_name, data_type)

    for i in tqdm(range(len(dataset)), desc="process... {} data".format(data_type)):
        paragraphs = dataset[i]['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context = context.replace("''", '" ').replace("``", '" ').replace('　', ' ')
            context_tokens, context_poses, context_ners, context_lemmas, context_tf = tokenize_pos_ner(context)
            # context = context.lower()  # 在tokenize后再小写，因为nltk对大小写敏感

            qas = paragraph['qas']
            charloc2wordloc = getcharloc2wordloc(context, context_tokens)  # char id 映射到 word id
            if charloc2wordloc is None:
                num_mappingprob += len(qas)
                continue

            for qa in qas:
                question = qa['question']
                uuid = qa['id']
                question_tokens, question_poses, question_ners, question_lemmas, _ = tokenize_pos_ner(question)

                question_tokens_lower = [w.lower() for w in question_tokens]
                context_tokens_lower = [w.lower() for w in context_tokens]
                exact_match = c2q_match(context_tokens,set(question_tokens))   # 精确匹配
                lower_match = c2q_match(context_tokens_lower, set(question_tokens_lower)) # 小写匹配
                lemma_match = c2q_match(context_lemmas, set(question_lemmas)) # 提取文章 token 的词干是否出现在问题中

                # 只选择第一个答案
                answer = qa['answers'][0]['text']
                ans_char_start = qa['answers'][0]['answer_start']
                ans_char_end = ans_char_start + len(answer)

                ans_word_start = charloc2wordloc[ans_char_start][1]
                ans_word_end = charloc2wordloc[ans_char_end - 1][1]

                answer_tokens = context_tokens[ans_word_start: ans_word_end + 1]  # 最后保存的是从context_tokens抽取的答案，并非原始提供的
                answer_span = [str(ans_word_start), str(ans_word_end)]  # 包括结束位置

                # ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                if ''.join(answer_tokens) != ''.join(answer.split()):
                    num_tokenprob += 1
                    # print(answer_tokens)
                    # print(answer.split())
                    continue

                # 将文章的数字特征合并保存
                feature = zip(context_poses, context_ners, context_tf, exact_match, lower_match, lemma_match)
                context_feature = [str(f) for f in feature]
                examples.append((SPLIT_TOKEN.join(context_tokens), SPLIT_TOKEN.join(question_tokens), SPLIT_TOKEN.join(answer_tokens),
                                 SPLIT_TOKEN.join(answer_span), SPLIT_TOKEN.join(context_feature), uuid))
    # 随机打乱
    indices = range(len(examples))
    np.random.shuffle(indices)

    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ",
          num_tokenprob)
    with codecs.open(os.path.join(out_dir, data_type) + '.context', 'w', encoding='utf-8') as context_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.question', 'w', encoding='utf-8') as question_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.answer', 'w', encoding='utf-8') as answer_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.span', 'w', encoding='utf-8') as answer_span_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.uuid', 'w', encoding='utf-8') as uuid_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.context_feature', 'w', encoding='utf-8') as context_feature_writer:

        for i in indices:
            context, question, answer, answer_span, context_feature, uuid = examples[i]
            context_writer.write(context.strip() + '\n')
            question_writer.write(question.strip() + '\n')
            answer_writer.write(answer.strip() + '\n')
            answer_span_writer.write(str(answer_span) + '\n')
            uuid_writer.write(uuid)
            context_feature_writer.write(context_feature.strip() + '\n')
    print("{}/{} examples saved".format(len(examples), len(examples) + num_tokenprob + num_mappingprob))


def main(FLAGS):
    raw_data_dir = FLAGS.raw_data_dir  # 原始json文件目录
    data_dir = FLAGS.data_dir  # 预处理后的数据存放目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_data = 'train-v1.1.json'
    dev_data = 'dev-v1.1.json'

    train_context_path = os.path.join(data_dir, "train.context")
    train_context_feature_path = os.path.join(data_dir, "train.context_feature")
    train_ques_path = os.path.join(data_dir, "train.question")
    train_ans_span_path = os.path.join(data_dir, "train.span")
    dev_context_path = os.path.join(data_dir, "dev.context")
    dev_context_feature_path = os.path.join(data_dir, "dev.context_feature")
    dev_ques_path = os.path.join(data_dir, "dev.question")
    dev_ans_span_path = os.path.join(data_dir, "dev.span")

    if not os.path.exists(train_context_path) or not os.path.exists(train_context_feature_path) or not os.path.exists(
            train_ques_path) or not os.path.exists(train_ans_span_path):
        preprocess_and_save_data('train', os.path.join(raw_data_dir, train_data), data_dir)

    if not os.path.exists(dev_context_path) or not os.path.exists(dev_context_feature_path) or not os.path.exists(
            dev_ques_path) or not os.path.exists(dev_ans_span_path):
        preprocess_and_save_data('dev', os.path.join(raw_data_dir, dev_data), data_dir)


if __name__ == '__main__':
    raw_data_dir = './data/raw/'
    train_data = 'train-v1.1.json'
    dev_data = 'dev-v1.1.json'
    output = './data/data/'  # 预处理的输出目录
    test = './data/test/'  # 测试用的临时输出目录

    preprocess_and_save_data('train', os.path.join(raw_data_dir, train_data), output)
    # preprocess_and_save_data('dev', os.path.join(raw_data_dir, dev_data), output)
