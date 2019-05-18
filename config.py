# -*- coding: utf8 -*-
import tensorflow as tf
import os
import time
import logging

flags = tf.flags

# 高级层面 选项
flags.DEFINE_integer("gpu", 1, "选择gpu")
# flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
flags.DEFINE_string("mode", "official_eval", "Available modes: train / show_examples / official_eval")

flags.DEFINE_string("experiment_name", "exp_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()),
                    "Unique name for your experiment. which will hold all data related to this experiment")
flags.DEFINE_integer("epochs", 20, "Number of epochs to train. 0 means train indefinitely")

# 超参数
flags.DEFINE_float("learning_rate", 0.001, "学习率")  # fusion 的时候请用 0.002
# flags.DEFINE_float("decay_steps", 100, "衰减步数")
# flags.DEFINE_float("decay_rate", 0.85, "衰减率")
flags.DEFINE_float("patience", 2, "dev_loss不下降的次数")
flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
flags.DEFINE_float("dropout", 0.15, " dropout")
flags.DEFINE_integer("batch_size", 32, "Batch size to use")

flags.DEFINE_integer("rnn_layer_num", 2, "定义多层 RNN")
flags.DEFINE_integer("pwnn_hidden_size", 256, "定义FFN的大小")
flags.DEFINE_integer("fusion_att_hidden_size", 256, "fusion attention 的维度")
flags.DEFINE_integer("hidden_size", 125, "Size of RNN layer")
flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states")  # 150 for bidaf ; #200 otherwise
flags.DEFINE_integer("hidden_size_qp_matching", 150, "Size of the hidden states")
flags.DEFINE_integer("hidden_size_sm_matching", 50, "Size of the hidden states")
flags.DEFINE_integer("hidden_size_fully_connected", 200, "Size of the hidden states")

flags.DEFINE_integer("context_len", 300, " 文章最长长度")
flags.DEFINE_integer("ques_len", 30, "问题最长长度")
flags.DEFINE_integer("word_len", 16, "单词最长长度")
flags.DEFINE_integer("embedding_size", 300, "预训练的词向量维度")
flags.DEFINE_integer("pos_embedding_size", 12, "pos embedding 的维度")
flags.DEFINE_integer("pos_nums", 57, "词性的种类")
flags.DEFINE_integer("ner_embedding_size", 10, "ner embedding 的维度")
flags.DEFINE_integer("ner_nums", 20, "命名实体的种类")

# char cnn 的参数
flags.DEFINE_integer("char_embedding_size", 8, "char embedding 的维度")
flags.DEFINE_integer("char_out_size", 100,
                     "num filters char CNN/out size")  # same as filer size; as suggested in handout
flags.DEFINE_integer("window_width", 5, "Kernel size for char cnn")  # as suggested in handout

# layer
flags.DEFINE_bool("add_char_embed", False, "Include char embedding -True/False")
flags.DEFINE_bool("add_highway_layer", True, "Add highway layer to concatenated embeddings -True/False")
flags.DEFINE_bool("rnet_attention", False, "Perform RNET QP and SM attention-True/False")
flags.DEFINE_bool("bidaf_attention", False, "Use BIDAF Attention-True/False")
flags.DEFINE_bool("answer_pointer_RNET", False, "Use Answer Pointer from RNET-True/False")
flags.DEFINE_bool("Chen", False, "Use Chen Danqi 的模型")
flags.DEFINE_bool("fusion", True, "Use fusion net 的模型")
flags.DEFINE_bool("bidaf_pointer", False, "Use bidaf_poiter")
flags.DEFINE_bool("answer_pointer", False, "Use Answer Pointer from RNET-True/False")
flags.DEFINE_bool("smart_span", True, "Select start and end idx based on smart conditions-True/False")

# 训练时保存，验证频率
flags.DEFINE_integer("print_every", 10, "多少 iterations 打印一次")
flags.DEFINE_integer("save_every", 500, "多少 iterations 保存一次模型")
flags.DEFINE_integer("eval_every", 500, "多少 iterations 计算验证集上的 loss/f1/em，这很耗时，不要太频繁")
flags.DEFINE_integer("keep", 1, "保存多少个 checkpoints， 0 表示保存全部 (这很占存储).")
flags.DEFINE_integer("log_level", logging.DEBUG, "日志打印级别")

# 因为服务器tf版本比较低，获取 FLAGS 的值后就不能再DEFINE_string，因此写到最后
FLAGS = flags.FLAGS  # 参数对象，可以通过点(.)获取参数
if FLAGS.mode != "train":
    FLAGS.batch_size = 512

# 文件位置
# MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = './'

DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据相关的根目录
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始json数据目录
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "data")  # 预处理后的数据目录
EMBED_DIR = os.path.join(ROOT_DIR, "embedding")  # glove以及处理后的embed目录

# 目录相关
MODEL_DIR = os.path.join(ROOT_DIR, "model", FLAGS.experiment_name)  # 模型保存根目录
CHECK_POINT_DIR = os.path.join(MODEL_DIR, "checkpoint")  # 模型保存目录
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_checkpoint")  # 最好的模型保存目录
SUMMARY_DIR = os.path.join(ROOT_DIR, "summary", FLAGS.experiment_name)  # tensorboard 目录
# 创建tensorboard文件夹
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
# 创建 模型保存 文件夹
if not os.path.exists(MODEL_DIR):
    os.makedirs(CHECK_POINT_DIR)
    os.makedirs(BEST_MODEL_DIR)

FLAGS.ckpt_path = CHECK_POINT_DIR,  # "Training directory to save the model parameters and other info."
FLAGS.best_model_ckpt_path = BEST_MODEL_DIR  # "Training directory to save the model parameters and other info."
FLAGS.embedding_dir = EMBED_DIR  # "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt"
FLAGS.embedding_file = "glove.840B.300d.txt"  # Where to find pretrained embeding for training."
FLAGS.raw_data_dir = RAW_DATA_DIR  # "Where to find raw SQuAD data for training. Defaults to data/"
FLAGS.data_dir = TRAIN_DATA_DIR  # "Where to find preprocessed SQuAD data for training. Defaults to data/"
FLAGS.summary_dir = SUMMARY_DIR  # "Where to find preprocessed SQuAD data for training. Defaults to data/"
FLAGS.predict_answer_file = "./data/prediction.json"  # 保存预测的答案

# 硬件,gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

# 提供为外部使用的配置信息
config = tf.flags.FLAGS
