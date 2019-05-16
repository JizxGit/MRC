# _*_ coding:utf8 _*_
import codecs
import numpy as np
import random
import time
from vocab import PAD_ID, UNK_ID
from vocab import get_word2id, get_tag2id, get_ner2id
import os
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
random.seed(42)
np.random.seed(42)
DATA_POOL_SIZE = 100
SPLIT_TOKEN = "  "


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, context_poses, context_ners, context_features, ques_ids, ques_mask, ques_tokens,
                 ans_span, ans_tokens, uuids=None):
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
        self.context_poses = context_poses
        self.context_ners = context_ners
        self.context_features = context_features  # 文章的特征（TF，准确匹配，小写匹配，词干匹配）

        self.ques_ids = ques_ids
        self.ques_mask = ques_mask
        self.ques_tokens = ques_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_poses)


def pad(token_batch, length=0, pad_id=0):
    '''
    将token_batch中的数据，pad成统一的长度
    :param token_batch: 要padd的一个batch数据
    :param length:  要pad成的长度,0表示pad成该batch中最长的list的长度
    :param pad_id: pad_id
    :return:
    '''

    pad_len = length
    if pad_len == 0:
        pad_len = max([len(token_list) for token_list in token_batch])

    pad_batch = [token_list + [pad_id] * (pad_len - len(token_list)) for token_list in token_batch]
    return pad_batch


# def split_by_whitespace(sentence):
#     words = []
#     for space_separated_fragment in sentence.strip().split():
#         words.extend(re.split(" ", space_separated_fragment))
#     return [w for w in words if w]


def sent2ids(text, word2id):
    tokens = text.split(SPLIT_TOKEN)
    ids = [word2id.get(token.lower(), UNK_ID) for token in tokens]
    return ids, tokens


def get_context_feature(context_feature_line, tag2id, ner2id):
    # context_feature_line的组成是：（context_poses, context_ners, context_tf, exact_match, lower_match, lemma_match）
    context_feature = context_feature_line.split(SPLIT_TOKEN)
    context_feature = [tuple(eval(f)) for f in context_feature]
    for i in context_feature:
        assert len(i) == 6
    context_pos_ids = [tag2id.get(f[0], 0) for f in context_feature]
    context_ner_ids = [ner2id.get(f[1], 0) for f in context_feature]
    context_tf_match = [[f[2], f[3], f[4], f[5]] for f in context_feature]

    return context_pos_ids, context_ner_ids, context_tf_match


def fill_batch_data_pool(batches, batch_size, context_len, ques_len, word2id, tag2id, ner2id, context_reader, context_feature_reader, ques_reader,
                         uuid_reader, ans_span_reader, truncate_long):
    # 从文件中读取到的以空格为划分的一行数据
    data_pool = []  # 数据池，当做缓冲区，用于打乱数据集

    while len(data_pool) < batch_size * DATA_POOL_SIZE:
        # 读取到的还是有大写的文本
        context_line = context_reader.readline()
        ques_line = ques_reader.readline()
        ans_span_line = ans_span_reader.readline()
        context_feature_line = context_feature_reader.readline()
        uuid = uuid_reader.readline().strip() #strip 避免微端的\n

        # 文件读取完了则退出
        if not (context_line and ques_line and ans_span_line and context_feature_line):
            break

        # 还是大写的 token
        context_ids, context_tokens = sent2ids(context_line, word2id)
        ques_ids, ques_tokens = sent2ids(ques_line, word2id)

        # 获取特征（文章的特征：pos，ner，TF，与问题准确匹配，小写匹配，词干匹配）
        context_pos_ids, context_ner_ids, context_tf_match = get_context_feature(context_feature_line, tag2id, ner2id)

        ans_span = [int(s) for s in ans_span_line.split()]
        if ans_span[1] < ans_span[0]:
            print("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue
        ans_tokens = context_tokens[ans_span[0]: ans_span[1] + 1]

        # 太长的训练数据，进行丢弃或者截取
        if len(context_ids) > context_len:
            if truncate_long:
                context_ids = context_ids[:context_len]
                context_pos_ids = context_pos_ids[:context_len]
                context_ner_ids = context_ner_ids[:context_len]
                context_tf_match = context_tf_match[:context_len]
            else:
                continue
        if len(ques_ids) > ques_len:
            if truncate_long:
                ques_ids = ques_ids[:ques_len]
            else:
                continue
        data_pool.append((uuid, context_ids, context_tokens, context_pos_ids, context_ner_ids, context_tf_match, ques_ids, ques_tokens, ans_span, ans_tokens))

    # 制作batch数据
    # random.shuffle(data_pool)
    for i in range(0, len(data_pool), batch_size):
        batch_uuids, batch_context_ids, batch_context_tokens, batch_context_pos_ids, batch_context_ner_ids, batch_context_tf_match, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens = zip(
            *data_pool[i: i + batch_size])
        batches.append((batch_uuids, batch_context_ids, batch_context_tokens, batch_context_pos_ids, batch_context_ner_ids, batch_context_tf_match, batch_ques_ids,
                        batch_ques_tokens, batch_ans_span, batch_ans_tokens))


def get_batch_data(config, data_type, word2id, truncate_long=False):
    '''
    提供数据的生成器
    :param config: 配置信息
    :param data_type:  需要提供的数据类型，可选的有：'train'\'dev'
    :param word2id:
    :param truncate_long: 截取长度超过限制的数据，在【训练时】一定直接丢弃该数据，不要截取，否则会出现错误
        因为end_label可能超过长度，sparse_softmax_cross_entropy_with_logits，就会因此label超出长度报错】，
        而在验证数据集上获取F1\EM时，不能丢弃数据，而要进行截取，因为预测的位置永远在[0,len]之间，end_label在这之外也没事，得到0分而已
    :return:
        Batch类型的对象，包含了一个batch_size的数据
    '''

    batch_size = config.batch_size
    context_len = config.context_len  # 文章的最长限制
    ques_len = config.ques_len  # 问题的最长限制
    tag2id = get_tag2id()
    ner2id = get_ner2id()

    context_file = os.path.join(config.data_dir, data_type + '.context')  # 文章文件地址
    context_feature_file = os.path.join(config.data_dir, data_type + '.context_feature')  # 文章的特征文件：【pos，ner，tf，exact-match，lower-match，lemma-match】
    ques_file = os.path.join(config.data_dir, data_type + '.question')  # 问题文件地址
    uuid_file = os.path.join(config.data_dir, data_type + '.uuid')  # 问题 的 uuid
    ans_span_file = os.path.join(config.data_dir, data_type + '.span')  # 答案span文件地址

    batch_data_pool = []  # batch_data的数据池，每次从中取出batch_size个数据
    with codecs.open(context_file, 'r', encoding='utf-8') as context_reader, \
            codecs.open(context_feature_file, 'r', encoding='utf-8') as context_feature_file_reader, \
            codecs.open(ques_file, 'r', encoding='utf-8') as ques_reader, \
            codecs.open(uuid_file, 'r', encoding='utf-8') as uuid_reader, \
            codecs.open(ans_span_file, 'r', encoding='utf-8') as ans_span_reader:
        while True:
            if len(batch_data_pool) == 0:  # 数据池空了，从文件中读取数据
                start = time.time()
                fill_batch_data_pool(batch_data_pool, batch_size, context_len, ques_len, word2id, tag2id, ner2id, context_reader,
                                     context_feature_file_reader, ques_reader, uuid_reader, ans_span_reader, truncate_long)
                end = time.time()
                # print('加载时间 : {}s'.format(end - start))
            if len(batch_data_pool) == 0:  # 填充后还是空，说明数据读取没了，退出
                break
            # 从数据池中拿一个batch_size的数据
            batch_uuids, batch_context_ids, batch_context_tokens, batch_context_pos_ids, batch_context_ner_ids, batch_context_features, batch_ques_ids, \
            batch_ques_tokens, batch_ans_span, batch_ans_tokens = batch_data_pool.pop(0)

            # 进行pad
            batch_context_ids = pad(batch_context_ids, context_len)
            batch_context_pos_ids = pad(batch_context_pos_ids, context_len)
            batch_context_ner_ids = pad(batch_context_ner_ids, context_len)
            batch_ques_ids = pad(batch_ques_ids, ques_len)
            batch_context_features = pad(batch_context_features, context_len, [0, 0, 0, 0])

            # np化
            batch_context_ids = np.asarray(batch_context_ids)
            batch_context_pos_ids = np.asarray(batch_context_pos_ids)
            batch_context_ner_ids = np.asarray(batch_context_ner_ids)
            batch_context_features = np.asarray(batch_context_features)
            batch_ques_ids = np.asarray(batch_ques_ids)
            batch_ans_span = np.asarray(batch_ans_span)

            # 进行mask，只有进行np化后，才能进行这样的操作
            batch_context_mask = (batch_context_ids != PAD_ID).astype(np.int32)
            batch_ques_mask = (batch_ques_ids != PAD_ID).astype(np.int32)

            batch = Batch(batch_context_ids, batch_context_mask, batch_context_tokens, batch_context_pos_ids, batch_context_ner_ids,
                          batch_context_features, batch_ques_ids, batch_ques_mask, batch_ques_tokens, batch_ans_span, batch_ans_tokens, batch_uuids)
            yield batch


if __name__ == '__main__':
    class Config():
        pass


    configg = Config()
    type = 'dev'
    GLOVE_DIR = './embedding/'
    configg.batch_size = 32
    configg.context_len = 300
    configg.ques_len = 30
    configg.data_dir = './data/data/'
    # _, w2id, _ = get_embedding_word2id_id2word(GLOVE_DIR + glove_file, 300)
    k = 0
    for batch in get_batch_data(configg, type, get_word2id(GLOVE_DIR + 'word2id.pickle')):
        print(batch.context_poses.shape)
        print(batch.context_ners.shape)
        print(batch.context_features.shape)
        print(batch.context_ids.shape)
        print(batch.uuids[0])
        print(' '.join(batch.ques_tokens[0]))
        print("----------")
        # k += batch.batch_size
        k += 1
        if k > 10:
            break
    print("size:", k)
