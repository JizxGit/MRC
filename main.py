# -*- coding: utf8 -*-
import tensorflow as tf
import os
import json
import codecs
from vocab import get_embedding_word2id_id2word
from preprocess import main as prepro
from model import Model
from evaluate import print_test_score

os.environ['CUDA_VISIBLE_DEVICES']='1'
# 高级层面 选项
tf.flags.DEFINE_integer("gpu", 1, "选择gpu")
tf.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.flags.DEFINE_string("experiment_name", "",
                       "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.flags.DEFINE_integer("epochs", 20, "Number of epochs to train. 0 means train indefinitely")

# 超参数
tf.flags.DEFINE_float("learning_rate", 0.01, "学习率")
tf.flags.DEFINE_float("decay_steps", 100, "衰减步数")
tf.flags.DEFINE_float("decay_rate", 0.85, "衰减率")
tf.flags.DEFINE_float("patience", 3, "dev_loss不下降的次数")
tf.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("dropout", 0.15, " dropout")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size to use")

tf.flags.DEFINE_integer("rnn_layer_num", 2, "定义多层 RNN")
tf.flags.DEFINE_integer("pwnn_hidden_size", 256, "定义FFN的大小")
tf.flags.DEFINE_integer("fusion_att_hidden_size", 256, "fusion attention 的维度")
tf.flags.DEFINE_integer("hidden_size", 125, "Size of RNN layer")
tf.flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states")  # 150 for bidaf ; #200 otherwise
tf.flags.DEFINE_integer("hidden_size_qp_matching", 150, "Size of the hidden states")
tf.flags.DEFINE_integer("hidden_size_sm_matching", 50, "Size of the hidden states")
tf.flags.DEFINE_integer("hidden_size_fully_connected", 200, "Size of the hidden states")

tf.flags.DEFINE_integer("context_len", 300, " 文章最长长度")
tf.flags.DEFINE_integer("ques_len", 30, "问题最长长度")
tf.flags.DEFINE_integer("word_len", 16, "单词最长长度")
tf.flags.DEFINE_integer("embedding_size", 300, "预训练的词向量维度")
tf.flags.DEFINE_integer("pos_embedding_size", 12, "pos embedding 的维度")
tf.flags.DEFINE_integer("pos_nums", 57, "词性的种类")
tf.flags.DEFINE_integer("ner_embedding_size", 10, "ner embedding 的维度")
tf.flags.DEFINE_integer("ner_nums", 20, "命名实体的种类")

# char cnn 的参数
tf.flags.DEFINE_integer("char_embedding_size", 8, "char embedding 的维度")
tf.flags.DEFINE_integer("char_out_size", 100, "num filters char CNN/out size")  # same as filer size; as suggested in handout
tf.flags.DEFINE_integer("window_width", 5, "Kernel size for char cnn")  # as suggested in handout

# layer
tf.flags.DEFINE_bool("add_char_embed", False, "Include char embedding -True/False")
tf.flags.DEFINE_bool("add_highway_layer", True, "Add highway layer to concatenated embeddings -True/False")
tf.flags.DEFINE_bool("rnet_attention", False, "Perform RNET QP and SM attention-True/False")
tf.flags.DEFINE_bool("bidaf_attention", False, "Use BIDAF Attention-True/False")
tf.flags.DEFINE_bool("answer_pointer_RNET", False, "Use Answer Pointer from RNET-True/False")
tf.flags.DEFINE_bool("Chen", False, "Use Chen Danqi 的模型")
tf.flags.DEFINE_bool("fusion", True, "Use fusion net 的模型")
tf.flags.DEFINE_bool("bidaf_pointer", False, "Use bidaf_poiter")
tf.flags.DEFINE_bool("answer_pointer", False, "Use Answer Pointer from RNET-True/False")
tf.flags.DEFINE_bool("smart_span", False, "Select start and end idx based on smart conditions-True/False")

# 训练时保存，验证频率
tf.flags.DEFINE_integer("print_every", 10, "多少 iterations 打印一次")
tf.flags.DEFINE_integer("save_every", 600, "多少 iterations 保存一次模型")
tf.flags.DEFINE_integer("eval_every", 600, "多少 iterations 计算验证集上的 loss/f1/em，这很耗时，不要太频繁")
tf.flags.DEFINE_integer("keep", 1, "保存多少个 checkpoints， 0 表示保存全部 (这很占存储).")

# 文件位置
# MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = './'

DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据相关的根目录
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始json数据目录
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "data")  # 预处理后的数据目录

EMBED_DIR = os.path.join(ROOT_DIR, "embedding")  # glove已经处理后的embed目录

MODEL_DIR = os.path.join(ROOT_DIR, "model")  # 模型保存根目录
CHECK_POINT_DIR = os.path.join(MODEL_DIR, "checkpoint")  # 模型保存目录
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_checkpoint")  # 最好的模型保存目录
SUMMARY_DIR = os.path.join(ROOT_DIR, "summary")  # tensorboard 目录

tf.flags.DEFINE_string("ckpt_path", CHECK_POINT_DIR, "Training directory to save the model parameters and other info.")
tf.flags.DEFINE_string("best_model_ckpt_path", BEST_MODEL_DIR, "Training directory to save the model parameters and other info.")
tf.flags.DEFINE_string("embedding_dir", EMBED_DIR, "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.flags.DEFINE_string("embedding_file", "", "Where to find pretrained embeding for training.")
tf.flags.DEFINE_string("raw_data_dir", RAW_DATA_DIR, "Where to find raw SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("data_dir", TRAIN_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("summary_dir", SUMMARY_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("predict_answer_file", "./data/prediction.json", "保存预测的答案")


# tf.flags.DEFINE_string("ckpt_load_dir", "","For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
# tf.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
# tf.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

def initial_model(session, ckpt_path, expect_exists=False):
    print("Looking for model at {}...".format(ckpt_path))
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at {}".format(ckpt_path))
        else:
            print('initial model...')
            session.run(tf.global_variables_initializer())


def main(unused_argv):
    config = tf.flags.FLAGS
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    # 创建必须的文件夹
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 处理embedding,word2id,id2word
    config.embedding_file = config.embedding_file or 'glove.6B.{}d.txt'.format(config.embedding_size)
    embed_matrix, word2id, id2word = get_embedding_word2id_id2word(config.embedding_dir, config.embedding_file, config.embedding_size)

    # 处理原始数据，保存处理后的数据
    prepro(config)

    # 创建模型
    qa_model = Model(config, embed_matrix, word2id, id2word)

    # 配置session信息 
    sess_config = tf.ConfigProto(allow_soft_placement=True)  # 是否打印设备分配日志;如果你指定的设备不存在,允许TF自动分配设备 
    sess_config.gpu_options.allow_growth = True  # 动态申请显存

    if config.mode == 'train':
        with tf.Session(config=sess_config) as sess:
            initial_model(sess, config.ckpt_path)
            qa_model.train(sess)

    elif config.mode == 'show_examples':
        pass
    elif config.mode == 'official_eval':
        with tf.Session(config=sess_config) as sess:
            # 1.获取最好的保存的模型
            initial_model(sess, config.best_model_ckpt_path, expect_exists=True)
            # 2.进行预测，保存预测结果
            uuid2ans = qa_model.test(sess)
            with codecs.open(config.predict_answer_file,'w',encoding='utf-8') as f:
                ans = unicode(json.dumps(uuid2ans, ensure_ascii=False))
                f.write(ans)
            # 3.评价
            print_test_score()

    else:
        raise Exception("未知的mode：「」".format(config.mode))


if __name__ == '__main__':
    tf.app.run()
