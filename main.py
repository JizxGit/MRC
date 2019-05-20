# -*- coding: utf8 -*-
import tensorflow as tf
import json
import codecs
from vocab import get_embedding_word2id_id2word
from preprocess import main as prepro
from config import config
from model import Model
from evaluate import print_test_score
import logging as log


def initial_model(session, ckpt_path, expect_exists=False):
    log.info("Looking for model at {}...".format(ckpt_path))
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at {}".format(ckpt_path))
        else:
            log.info('initial model...')
            session.run(tf.global_variables_initializer())


def main(unused_argv):
    # 处理embedding,word2id,id2word
    config.embedding_file = config.embedding_file or 'glove.6B.{}d.txt'.format(config.embedding_size)
    embed_matrix, word2id, id2word = get_embedding_word2id_id2word(config.embedding_dir, config.embedding_file,
                                                                   config.embedding_size)
    # 处理原始数据，保存处理后的数据
    prepro(config)

    # 创建模型
    qa_model = Model(config, embed_matrix, word2id, id2word)

    # 配置session信息 
    sess_config = tf.ConfigProto(allow_soft_placement=True)  # 是否打印设备分配日志;如果你指定的设备不存在,允许TF自动分配设备 
    sess_config.gpu_options.allow_growth = True  # 动态申请显存

    log.info("### 『{}』 model is working in『{}』 mode，batch_size：『{}』###".format(config.experiment_name, config.mode, config.batch_size))
    if config.mode == 'train':
        with tf.Session(config=sess_config) as sess:
            initial_model(sess, config.ckpt_path)
            qa_model.train(sess)

    elif config.mode == 'show_examples':
        with tf.Session(config=sess_config) as sess:
            # 1.获取最好的保存的模型
            initial_model(sess, config.best_model_ckpt_path, expect_exists=True)
            # 2.进行预测，保存预测结果
            dev_loss, dev_f1, dev_em = qa_model.validate(sess)
            print(dev_loss, dev_f1, dev_em)

    elif config.mode == 'official_eval':
        with tf.Session(config=sess_config) as sess:
            # 1.获取最好的保存的模型
            initial_model(sess, config.best_model_ckpt_path, expect_exists=True)
            # 2.进行预测，保存预测结果
            uuid2ans = qa_model.test(sess)
            with codecs.open(config.predict_answer_file, 'w', encoding='utf-8') as f:
                ans = unicode(json.dumps(uuid2ans, ensure_ascii=False))
                f.write(ans)
            # 3.评价
            print_test_score()
    else:
        raise Exception("未知的mode：{}".format(config.mode))


if __name__ == '__main__':
    tf.app.run()
