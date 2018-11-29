# _*_ coding:utf8 _*_
from tqdm import tqdm
import numpy as np
import codecs
import os
import pickle
import time

PAD_ID = 0
UNK_ID = 1
_PAD = u"<pad>"
_UNK = u"<unk>"
SPECIAL_TOKEN = [_PAD, _UNK]


def load_file(file_name, msg=None):
    if msg is not None:
        print('加载 {} ...'.format(msg))
    with open(file_name, 'rb')  as f:
        return pickle.load(f)


def save_file(file, file_name, msg=None):
    if msg is not None:
        print('保存 {} ...'.format(msg))
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)


def get_embedding_word2id_id2word(embedding_path, pretrained_embed_filename, dim, vocab_size=int(4e5), reuse=True):
    '''
    处理glove 获取词向量矩阵，word2id，id2word
    :param embedding_path:  存放预训练的词向量的目录
    :param pretrained_embed_filename: 预训练好的词向量名称
    :param dim: 词向量维度
    :param vocab_size: 词典单词个数
    :param reuse: 是否使用已经保存的文件
    :return:
    '''

    start = time.time()
    embed_matrix_file = os.path.join(embedding_path, "embed_matrix.pickle")
    word2id_file = os.path.join(embedding_path, "word2id.pickle")
    id2word_file = os.path.join(embedding_path, "id2word.pickle")
    embedding_file=os.path.join(embedding_path, pretrained_embed_filename)

    embed_matrix = np.zeros([vocab_size + len(SPECIAL_TOKEN), dim], dtype=np.float32)
    word2id = {}
    id2word = {}

    # 如果之前处理过文件就直接加载、返回
    if reuse and os.path.exists(embed_matrix_file) and os.path.exists(word2id_file) and os.path.exists(id2word_file):
        print('加载预处理好的词向量')
        embed_matrix = load_file(embed_matrix_file, 'embed_matrix')
        word2id = load_file(word2id_file, 'word2id')
        id2word = load_file(id2word_file, 'id2word')

    else:
        # 重新处理
        print('处理 预训练的词向量')
        id = 0
        for word in SPECIAL_TOKEN:
            word2id[word] = id
            id2word[id] = word
            id += 1
        embed_matrix[1, :] = np.random.randn(1, dim)

        with codecs.open(embedding_file, 'r', encoding='utf-8') as f:
            for _ in tqdm(xrange(vocab_size), desc="processing {}d glove".format(dim)):
                line = f.readline().strip().split(" ")
                word = line[0]
                vector = [float(v) for v in line[1:]]
                assert len(vector) == dim
                id2word[id] = word
                word2id[word] = id
                embed_matrix[id, :] = vector
                id += 1

        save_file(embed_matrix, embed_matrix_file, 'embed_matrix')
        save_file(word2id, word2id_file, 'word2id')
        save_file(id2word, id2word_file, 'id2word')

    end = time.time()
    print('加载时间 : {}s'.format(end - start))
    return embed_matrix, word2id, id2word


def get_word2id(word2id_file):
    word2id = load_file(word2id_file, 'word2id')
    return word2id


if __name__ == "__main__":
    glove_file = "glove.6B.300d.txt"
    em,_,_=get_embedding_word2id_id2word('./embedding/',glove_file,300)
    print(em[0])
    print(em[2])
    print(em[18])
