# -*- coding: utf8 -*-
import tensorflow as tf
import os
import time
import logging

######################## flags 与 tf 的版本有关 ########################
# 服务器tf.v4
# tf.v14 的可以边获取Flags.var，边定义新的参数，但是不支持Flags.var=num 进行直接赋值，之前必须DEFINE_string（var，""）过
# 服务器的flag，可以进行获取(设置)参数 Flags.var ，但是你接下来定义的tf.flags.DEFINE_float将会出现AttributeError，
# 但是如果导入到其他py 中就可以获取,或者可以Flags.var=num 进行直接赋值


# 高级层面 选项
tf.flags.DEFINE_integer("gpu", 1, "选择gpu")
tf.flags.DEFINE_string("mode", "official_eval", "Available modes: train / show_examples / official_eval")
tf.flags.DEFINE_string("experiment_name", "exp_" + time.strftime("%Y%m%d_%H%M", time.localtime()),
                       "Unique name for your experiment. which will hold all data related to this experiment")
tf.flags.DEFINE_integer("epochs", 12, "Number of epochs to train. 0 means train indefinitely")

# 超参数
tf.flags.DEFINE_float("learning_rate", 0.001, "学习率")  # fusion 的时候请用 0.002
# tf.flags.DEFINE_float("decay_steps", 100, "衰减步数")
# tf.flags.DEFINE_float("decay_rate", 0.85, "衰减率")
tf.flags.DEFINE_float("patience", 3, "dev_loss不下降的次数超过patience，就将学习率减半")
tf.flags.DEFINE_float("exit_threshold", 9, "dev_loss不下降的次数超过 exit_threshold，停止训练")
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
tf.flags.DEFINE_integer("char_out_size", 100,
                        "num filters char CNN/out size")  # same as filer size; as suggested in handout
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
tf.flags.DEFINE_bool("smart_span", True, "Select start and end idx based on smart conditions-True/False")

# 训练时保存，验证频率
tf.flags.DEFINE_integer("print_every", 10, "多少 iterations 打印一次")
tf.flags.DEFINE_integer("save_every", 500, "多少 iterations 保存一次模型")
tf.flags.DEFINE_integer("eval_every", 500, "多少 iterations 计算验证集上的 loss/f1/em，这很耗时，不要太频繁")
tf.flags.DEFINE_integer("keep", 1, "保存多少个 checkpoints， 0 表示保存全部 (这很占存储).")
tf.flags.DEFINE_integer("log_level", logging.DEBUG, "日志打印级别")

ROOT_DIR = './'
DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据相关的根目录
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始json数据目录
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "data")  # 预处理后的数据目录
EMBED_DIR = os.path.join(ROOT_DIR, "embedding")  # glove以及处理后的embed目录

# 目录相关
FLAGS = tf.flags.FLAGS  # 参数对象，可以通过点(.)获取参数
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

# 文件位置

tf.flags.DEFINE_string("ckpt_path", CHECK_POINT_DIR, "Training directory to save the model parameters and other info.")
tf.flags.DEFINE_string("best_model_ckpt_path", BEST_MODEL_DIR,
                       "Training directory to save the model parameters and other info.")
tf.flags.DEFINE_string("embedding_dir", EMBED_DIR,
                       "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.flags.DEFINE_string("embedding_file", "glove.840B.300d.txt", "Where to find pretrained embeding for training.")
tf.flags.DEFINE_string("raw_data_dir", RAW_DATA_DIR, "Where to find raw SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("data_dir", TRAIN_DATA_DIR,
                       "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("summary_dir", SUMMARY_DIR,
                       "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.flags.DEFINE_string("predict_answer_file", "./data/prediction.json", "保存预测的答案")

#### web.py 兼容用的，不知道什么原因，导入 main.py就是可以用，导入 web.py 就是不行
FLAGS.ckpt_path = CHECK_POINT_DIR # "Training directory to save the model parameters and other info."
FLAGS.best_model_ckpt_path = BEST_MODEL_DIR  # "Training directory to save the model parameters and other info."
FLAGS.embedding_dir = EMBED_DIR  # "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt"
FLAGS.embedding_file = "glove.840B.300d.txt"  #"Where to find pretrained embeding for training."
FLAGS.raw_data_dir = RAW_DATA_DIR  # "Where to find raw SQuAD data for training. Defaults to data/"
FLAGS.data_dir = TRAIN_DATA_DIR  # "Where to find preprocessed SQuAD data for training. Defaults to data/"
FLAGS.summary_dir = SUMMARY_DIR  # Where to find preprocessed SQuAD data for training. Defaults to data/"
FLAGS.predict_answer_file = "./data/prediction.json"  # 保存预测的答案"

# 因为服务器tf版本比较低，设置 FLAGS 的值后就不能再DEFINE_string，因此写到最后
if FLAGS.mode != "train":
    FLAGS.batch_size = 512

# 硬件,gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

# 提供为外部使用的配置信息
config = tf.flags.FLAGS
