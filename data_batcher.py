# _*_ coding:utf8 _*_
import codecs
import numpy as np
import random
import re
from vocab import PAD_ID, UNK_ID
from vocab import get_word2id, get_tag2id, get_ner2id

from collections import Counter
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

DATA_POOL_SIZE = 128


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, ques_ids, ques_mask, ques_tokens, ans_span, ans_tokens,
                 uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          context_feature : 文章的特征（准确匹配，小写匹配，词干匹配，TF）
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens
        # self.context_poses = context_poses
        # self.context_ners = context_ners
        # self.context_tfs = context_tfs
        # self.context_feature = context_feature  # 文章的特征（准确匹配，小写匹配，词干匹配，TF）

        self.ques_ids = ques_ids
        self.ques_mask = ques_mask
        self.ques_tokens = ques_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


def process_context(context):
    c_doc = spacy_nlp(re.sub(r'\s', ' ', context))
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]


def pad(token_batch, length=0):
    '''
    将token_batch中的数据，pad成统一的长度
    :param token_batch: 要padd的一个batch数据
    :param length:  要pad成的长度,0表示pad成该batch中最长的list的长度
    :return:
    '''

    pad_len = length
    if pad_len == 0:
        pad_len = max([len(token_list) for token_list in token_batch])

    pad_batch = [token_list + [PAD_ID] * (pad_len - len(token_list)) for token_list in token_batch]
    return pad_batch


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def sent2ids(tokens, word2id):
    ids = [word2id.get(token.lower(), UNK_ID) for token in tokens]
    return ids


def get_context_feature(c_doc, q_doc, context_tokens, question_tokens):
    # 1、文章自身的特征（POS、NER、TF）
    # spacy的说明 https://spacy.io/usage/linguistic-features
    # nltk.pos_tag(context_tokens)
    # nltk.ne_chunk()
    context_pos = [w.tag_ for w in c_doc]  # 获取 POS(pos_: The simple part-of-speech tag; Tag_: The detailed part-of-speech tag.)
    context_ner = [w.ent_type_ for w in c_doc]  # 获取token 级别的 NER （命名实体识别）

    # 计算 TF
    context_tokens_lower = [w.lower() for w in context_tokens]
    counter = Counter(context_tokens_lower)
    total = len(context_tokens)
    context_tf = [counter[w] / total for w in context_tokens_lower]

    # 2.与问题相关的特征：精准匹配、小写匹配、词干匹配（match_origin、match_lower、match_lemma）
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set([w.lower() for w in question_tokens])
    question_lemma_set = set([w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc])

    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma_set for w in c_doc]

    # 下面这种写法 是 这样断句的：[ w.lemma_  if w.lemma_ != '-PRON-' else  (w.text.lower() in question_lemma_set) for w in c_doc ]
    # 也就是说，要是 w.lemma_ 要么是 True、False
    # match_lemma = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() in question_lemma_set for w in c_doc]
    return context_pos, context_ner, context_tf, match_origin, match_lower, match_lemma


def fill_batch_data_pool(batches, batch_size, context_len, ques_len, word2id, tag2id, ner2id, context_reader, context_pos_reader, context_ner_reader,
                         context_tf_reader, context_exact_match_reader, context_lower_match_reader, context_lemma_match_reader, ques_reader,
                         ans_span_reader, truncate_long=True):
    # 从文件中读取到的以空格为划分的一行数据

    data_pool = []  # 数据池，当做缓冲区，用于打乱数据集

    while len(data_pool) < batch_size * DATA_POOL_SIZE:
        # 读取到的还是有大写的文本
        context_line = context_reader.readline()
        ques_line = ques_reader.readline()
        ans_span_line = ans_span_reader.readline()
        # 文件读取完了则退出
        if not (context_line and ques_line and ans_span_line):
            break

        # 使用 spacy_nlp 进行处理
        c_doc = spacy_nlp(context_line)
        q_doc = spacy_nlp(ques_line)

        context_tokens = [w.text for w in c_doc]
        context_tokens_=context_line.split(" ")
        if len(context_tokens)!=len(context_tokens_):
            print(context_line)
            exit()
        ques_tokens = [w.text for w in q_doc]

        # sentence to id, 还是大写的 token
        context_ids = sent2ids(context_tokens, word2id)
        ques_ids = sent2ids(ques_tokens, word2id)

        # 获取特征（文章的特征：与问题准确匹配，小写匹配，词干匹配，TF）
        context_pos, context_ner, context_tf, match_origin, match_lower, match_lemma = get_context_feature(c_doc, q_doc, context_tokens, ques_tokens)
        print(len(data_pool))

        ans_span = [int(s) for s in ans_span_line.split()]
        if ans_span[1] < ans_span[0]:
            print("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue
        ans_tokens = context_tokens[ans_span[0]: ans_span[1] + 1]

        # 太长的训练数据，进行丢弃或者截取
        if len(context_ids) > context_len:
            if truncate_long:
                context_ids = context_ids[:context_len]
            else:
                continue
        if len(ques_ids) > ques_len:
            if truncate_long:
                ques_ids = ques_ids[:ques_len]
            else:
                continue

        data_pool.append((context_ids, context_tokens, context_pos, context_ner, context_tf, match_origin, match_lower, match_lemma, ques_ids,
                          ques_tokens, ans_span, ans_tokens))

    # 制作batch数据
    random.shuffle(data_pool)

    for i in xrange(0, len(data_pool), batch_size):
        batch_content_ids, batch_content_tokens, batch_context_pos, batch_context_ner, batch_context_tf, batch_match_origin, batch_match_lower, batch_match_lemma, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens = zip(
            *data_pool[i: i + batch_size])
        batches.append((batch_content_ids, batch_content_tokens, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens))

    random.shuffle(batches)


def get_batch_data(config, data_type, word2id, truncate_long=False):
    '''
    提供数据的生成器
    :param context_file: 文章文件地址
    :param ques_file: 问题文件地址
    :param ans_span_file: 答案span文件地址
    :param batch_size: batch_size
    :param word2id: word2id
    :param context_len: 文章的最长限制
    :param ques_len: 问题的最长限制
    :param truncate_long:  截取长度超过限制的数据，在【训练时】一定直接丢弃该数据，不要截取，否则会出现错误
        因为end_label可能超过长度，sparse_softmax_cross_entropy_with_logits，就会因此label超出长度报错】，
        而在验证数据集上获取F1\EM时，不能丢弃数据，而要进行截取，因为预测的位置永远在[0,len]之间，end_label在这之外也没事，得到0分而已
    :return:
        Batch类型的对象，包含了一个batch_size的数据
    '''

    batch_size = config.batch_size
    context_len = config.context_len
    ques_len = config.ques_len
    tag2id = get_tag2id()
    ner2id = get_ner2id()

    context_file = config.prepro_data_dir + data_type + '.context'
    context_pos_file = config.prepro_data_dir + data_type + '.pos'
    context_ner_file = config.prepro_data_dir + data_type + '.ner'
    context_tf_file = config.prepro_data_dir + data_type + '.tf'
    context_exact_match_file = config.prepro_data_dir + data_type + '.exact_match'
    context_lower_match_file = config.prepro_data_dir + data_type + '.lower_match'
    context_lemma_match_file = config.prepro_data_dir + data_type + '.lemma_match'
    ques_file = config.prepro_data_dir + data_type + '.question'
    ans_span_file = config.prepro_data_dir + data_type + '.span'

    batch_data_pool = []  # batch_data的数据池，每次从中取出batch_size个数据
    with codecs.open(context_file, 'r', encoding='utf-8') as context_reader, \
            codecs.open(context_pos_file, 'r', encoding='utf-8') as context_pos_reader, \
            codecs.open(context_ner_file, 'r', encoding='utf-8') as context_ner_reader, \
            codecs.open(context_tf_file, 'r', encoding='utf-8') as context_tf_reader, \
            codecs.open(context_exact_match_file, 'r', encoding='utf-8') as context_exact_match_reader, \
            codecs.open(context_lower_match_file, 'r', encoding='utf-8') as context_lower_match_reader, \
            codecs.open(context_lemma_match_file, 'r', encoding='utf-8') as context_lemma_match_reader, \
            codecs.open(ques_file, 'r', encoding='utf-8') as ques_reader, \
            codecs.open(ans_span_file, 'r', encoding='utf-8') as ans_span_reader:
        while True:
            if len(batch_data_pool) == 0:  # 数据池空了，从文件中读取数据
                fill_batch_data_pool(batch_data_pool, batch_size, context_len,
                                     ques_len, word2id, tag2id, ner2id, context_reader, context_pos_reader, context_ner_reader, context_tf_reader,
                                     context_exact_match_reader, context_lower_match_reader, context_lemma_match_reader, ques_reader, ans_span_reader,
                                     truncate_long)
                if len(batch_data_pool) == 0:  # 填充后还是空，说明数据读取没了，退出
                    break

                batch_context_ids, batch_context_tokens, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens = batch_data_pool.pop(
                    0)  # 从数据池中拿一个batch_size的数据

                # 进行pad
                batch_context_ids = pad(batch_context_ids, context_len)
                batch_ques_ids = pad(batch_ques_ids, ques_len)

                # np化
                batch_context_ids = np.array(batch_context_ids)
                batch_ques_ids = np.array(batch_ques_ids)
                batch_ans_span = np.array(batch_ans_span)

                # 进行mask,最有进行np化后，才能进行这样的操作
                batch_context_mask = (batch_context_ids != PAD_ID).astype(np.int32)
                batch_ques_mask = (batch_ques_ids != PAD_ID).astype(np.int32)

                batch = Batch(batch_context_ids, batch_context_mask, batch_context_tokens, batch_ques_ids, batch_ques_mask, batch_ques_tokens,
                              batch_ans_span, batch_ans_tokens)
                yield batch


if __name__ == '__main__':
    class Config():
        pass

    config = Config()
    type = 'dev'
    GLOVE_DIR = './embedding/'
    config.batch_size = 32
    config.context_len = 300
    config.ques_len = 30
    config.prepro_data_dir = './data/data/'
    # _, w2id, _ = get_embedding_word2id_id2word(GLOVE_DIR + glove_file, 300)
    i = 0
    for batch in get_batch_data(config, type, get_word2id(GLOVE_DIR + 'word2id.pickle')):
        i += batch.batch_size
        # for tokens in batch.context_tokens:
        # print(' '.join(tokens))
        # print(i)
    print(i)
