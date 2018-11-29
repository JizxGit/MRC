# _*_ coding:utf8 _*_
import codecs
import numpy as np
import random
import re
from vocab import PAD_ID, UNK_ID
from vocab import get_word2id

DATA_POOL_SIZE = 128


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, ques_ids, ques_mask, ques_tokens, ans_span, ans_tokens, uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.ques_ids = ques_ids
        self.ques_mask = ques_mask
        self.ques_tokens = ques_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


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


def sent2ids(sent, word2id):
    # tokens = sent.split(' ')
    # TODO 这真的有影响吗？
    tokens = split_by_whitespace(sent)
    ids = [word2id.get(token, UNK_ID) for token in tokens]
    return tokens, ids


def fill_batch_data_pool(batches, batch_size, word2id, context_reader, ques_reader, ans_span_reader, context_len, ques_len, truncate_long=True):
    # 从文件中读取到的以空格为划分的一行数据

    data_pool = []  # 数据池，当做缓冲区，用于打乱数据集

    while len(data_pool) < batch_size * DATA_POOL_SIZE:
        context_line, ques_line, ans_span_line = context_reader.readline(), ques_reader.readline(), ans_span_reader.readline()
        if not (context_line and ques_line and ans_span_line):
            break

        context_tokens, context_ids = sent2ids(context_line, word2id)
        ques_tokens, ques_ids = sent2ids(ques_line, word2id)
        ans_span = [int(s) for s in ans_span_line.split()]

        assert len(ans_span) == 2
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

        data_pool.append((context_ids, context_tokens, ques_ids, ques_tokens, ans_span, ans_tokens))
        # 训练数据满了，则退出循环，进行shuffle
        # if len(data_pool) == batch_size * DATA_POOL_SIZE:
        #     break

        # context_line, ques_line, ans_span_line = context_reader.readline(), ques_reader.readline(), ans_span_reader.readline()

    # 制作batch数据
    random.shuffle(data_pool)
    # data_pool = sorted(data_pool,key=lambda e:len(e[2])) # 打乱数据？

    for i in xrange(0, len(data_pool), batch_size):
        batch_content_ids, batch_content_tokens, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens = zip(
            *data_pool[i: i + batch_size])
        batches.append((batch_content_ids, batch_content_tokens, batch_ques_ids, batch_ques_tokens, batch_ans_span, batch_ans_tokens))

    random.shuffle(batches)

def get_batch_data(context_file_path, ques_file_path, ans_span_file_path, batch_size, word2id, context_len, ques_len, truncate_long=False):
    '''
    提供数据的生成器
    :param context_file_path: 文章文件地址
    :param ques_file_path: 问题文件地址
    :param ans_span_file_path: 答案span文件地址
    :param batch_size: batch_size
    :param word2id: word2id
    :param context_len: 文章的最长限制
    :param ques_len: 问题的最长限制
    :param truncate_long:  截取长度超过限制的数据，在训练时一定不要截取而是直接丢弃该数据，否则会出现错误【
        因为end_label可能超过长度，sparse_softmax_cross_entropy_with_logits，就会因此label超出长度报错】，
        而在验证数据集上获取F1\EM时，不能丢弃数据，而要进行截取，因为预测的位置永远在[0,len]之间，end_label在这之外也没事，得到0份而已
    :return:
        Batch类型的对象，包含了一个batch_size的数据
    '''

    batch_data_pool = []  # batch_data的数据池，每次从中取出batch_size个数据
    with codecs.open(context_file_path, 'r', encoding='utf-8') as context_reader, \
            codecs.open(ques_file_path, 'r', encoding='utf-8') as ques_reader, \
            codecs.open(ans_span_file_path, 'r', encoding='utf-8') as ans_span_reader:
        while True:
            if len(batch_data_pool) == 0:  # 数据池空了，从文件中读取数据
                fill_batch_data_pool(batch_data_pool, batch_size, word2id, context_reader, ques_reader, ans_span_reader, context_len, ques_len,
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
    root = './data/data/'
    type = 'dev'
    cf = root + type + '.context'
    qf = root + type + '.question'
    af = root + type + '.span'
    glove_file = "glove.6B.300d.txt"
    GLOVE_DIR = './embedding/'
    # _, w2id, _ = get_embedding_word2id_id2word(GLOVE_DIR + glove_file, 300)
    i=0
    for batch in get_batch_data(cf, qf, af, 32, get_word2id(GLOVE_DIR + 'word2id.pickle'), 300, 30):
        for tokens in batch.context_tokens:
            # print(' '.join(tokens))
            i+=1
    print(i)
