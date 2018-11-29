# coding=utf-8
import numpy as np
import tensorflow as tf
import os
import time
from modules import *
from data_batcher import get_batch_data
from evaluate import f1_score, em_score
from tensorflow.python.ops import variable_scope as vs


class Model(object):
    def __init__(self, FLAGS, embed_matrix, word2id, id2word):
        self.FLAGS = FLAGS
        self.word2id = word2id
        self.id2word = id2word
        # char的个数
        char2id, id2char, num_chars = self.create_char_dict()
        self.char2id = char2id
        self.id2char = id2char
        self.num_chars = num_chars

        # 构建图
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholder()
            self.add_embedding(embed_matrix)
            self.build_graph()
            self.add_loss()

        # 梯度剪切
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # train op
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # 定义 savers (for checkpointing) 和 summaries (for tensorboard)
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        # self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

    def add_placeholder(self):
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.ques_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len])
        self.ques_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # 默认的值
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        # char cnn
        if self.FLAGS.add_char_embed:
            self.char_ids_context = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])
            self.char_ids_ques = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len, self.FLAGS.word_len])

    def add_embedding(self, embed_matrix):
        with vs.variable_scope("embedding"):
            embeding_matrix = tf.constant(embed_matrix, dtype=tf.float32, name="word_embed_matrix")
            self.context_emb = tf.nn.embedding_lookup(embeding_matrix, self.context_ids)
            self.ques_emb = tf.nn.embedding_lookup(embeding_matrix, self.ques_ids)

    def create_char_dict(self):
        unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                        '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                        'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        '~', ]
        CHAR_PAD_ID = 0
        CHAR_UNK_ID = 1
        _CHAR_PAD = '*'
        _CHAR_UNK = '@'

        num_chars = len(unique_chars)
        id2char = {id: char for id, char in enumerate(unique_chars, 2)}
        id2char[CHAR_PAD_ID] = _CHAR_PAD
        id2char[CHAR_UNK_ID] = _CHAR_UNK
        char2id = {char: id for id, char in id2char.iteritems()}
        return char2id, id2char, num_chars

    def add_char_embedding(self):

        def conv1d(inputs, output_size, width, stride, scope_name):
            length = inputs.get_shape()[-1]  # char embedding_size
            inputs = tf.expand_dims(inputs, axis=3)  # 增加channel 维度
            with vs.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                filter = tf.get_variable("filter", [width, length, 1, output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))

            convolved = tf.nn.conv2d(inputs, filter=filter, strides=[1, stride, 1, 1], padding='VALID')
            print("conv2d:", convolved.shape)
            result = tf.squeeze(convolved, axis=2)  # 在squeeze之前：[batch_size, word_len-width, 1, output_size]
            print("conv2d result:", result.shape)
            return result

        # 可训练的矩阵，dtype默认为float32
        char_embed_matrix = tf.Variable(tf.random_uniform([self.num_chars, self.FLAGS.char_embedding_size], -1, 1))

        self.context_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_context, shape=(-1, self.FLAGS.word_len)))
        self.context_char_emb = tf.reshape(self.context_char_emb, shape=(-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size))
        print("context_char_emb :", self.context_char_emb.shape)
        # shape = batch_size * context_len, word_max_len, char_embedding_size

        self.ques_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_ques, shape=(-1, self.FLAGS.word_len)))
        self.ques_char_emb = tf.reshape(self.ques_char_emb, shape=(-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size))
        print("ques_char_emb :", self.context_char_emb.shape)

        # char CNN
        # ############# context ##############
        self.context_cnn_out = conv1d(inputs=self.context_char_emb, output_size=self.FLAGS.char_out_size, width=self.FLAGS.window_width, stride=1,
                                      scope_name='char-cnn')
        self.context_cnn_out = tf.nn.dropout(self.context_cnn_out, self.keep_prob)
        # shape= [batch*context_len,word_len-witdh,char_out_size]
        print("Shape context embs after conv", self.context_cnn_out.shape)

        self.context_cnn_out = tf.reduce_sum(self.context_cnn_out, axis=1)
        self.context_cnn_out = tf.reshape(self.context_cnn_out, shape=(
            -1, self.FLAGS.context_len, self.FLAGS.char_out_size))
        # shape= [batch  context_len, char_out_size]
        print("Shape context embs after pooling", self.context_cnn_out.shape)

        # ############# question #############
        self.ques_cnn_out = conv1d(inputs=self.ques_char_emb, output_size=self.FLAGS.char_out_size, width=self.FLAGS.window_width, stride=1,
                                   scope_name='char-cnn')
        self.ques_cnn_out = tf.nn.dropout(self.ques_cnn_out, self.keep_prob)
        print("Shape ques embs after conv", self.ques_cnn_out.shape)

        self.ques_cnn_out = tf.reduce_sum(self.ques_cnn_out, axis=1)
        self.ques_cnn_out = tf.reshape(self.ques_cnn_out, shape=(-1, self.FLAGS.ques_len, self.FLAGS.char_out_size))
        print("Shape ques embs after pooling", self.ques_cnn_out.shape)

        return self.context_cnn_out, self.ques_cnn_out

    def build_graph(self):

        ######################################  Char Embedding  ######################################
        if self.FLAGS.add_char_embed:
            self.context_char_out, self.ques_char_out = self.add_char_embedding()
            # 将word embed和 char embed 拼接
            self.context_emb = tf.concat([self.context_emb, self.context_char_out], axis=2)
            print("Shape - concatenated context embs", self.context_emb.shape)
            self.ques_emb = tf.concat([self.ques_emb, self.ques_char_out], axis=2)
            print("Shape - concatenated qn embs", self.ques_emb.shape)

        ######################################  highway   ######################################
        if self.FLAGS.add_highway_layer:
            emb_size = self.context_emb.get_shape().as_list()[-1]
            for _ in range(2):
                self.context_emb = self.highway(self.context_emb, emb_size, scope_name='highway')
                self.ques_emb = self.highway(self.ques_emb, emb_size, scope_name='highway')
            pass

        ######################################  RNNEncoder  ######################################
        encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_emb, self.context_mask, scope_name="RNNEncoder")
        ques_hiddens = encoder.build_graph(self.ques_emb, self.ques_mask, scope_name="RNNEncoder")

        ######################################  Basic Attention  ######################################
        # last_dim=context_hiddens.shape().as_list()[-1]
        if self.FLAGS.rnet_attention:  ##perform Question Passage and Self Matching attention from R-Net

            rnet_layer = Attention_Match_RNN(self.keep_prob, self.FLAGS.hidden_size_encoder, self.FLAGS.hidden_size_qp_matching,
                                             self.FLAGS.hidden_size_sm_matching)

            # Implement better question_passage matching
            v_P = rnet_layer.build_graph_qp_matching(context_hiddens, ques_hiddens, self.ques_mask, self.context_mask, self.FLAGS.context_len,
                                                     self.FLAGS.ques_len)

            self.rnet_attention = v_P

            # self.rnet_attention = tf.squeeze(self.rnet_attention, axis=[2])  # shape (batch_size, seq_len)

            # Take softmax over sequence
            # _, self.rnet_attention_probs = masked_softmax(self.rnet_attention, self.context_mask, 1)

            h_P = rnet_layer.build_graph_sm_matching(context_hiddens, ques_hiddens, self.ques_mask, self.context_mask,
                                                     self.FLAGS.context_len, self.FLAGS.ques_len, v_P)

            # Blended reps for R-Net
            blended_represent = tf.concat([context_hiddens, v_P, h_P], axis=2)  # (batch_size, context_len, hidden_size*6)
            print("Blended reps for R-Net", blended_represent.shape)
        elif self.FLAGS.bidaf_attention:

            # attention 层
            attn_layer = Bidaf(self.keep_prob, self.FLAGS.hidden_size_encoder * 2)
            attn_output = attn_layer.build_graph(context_hiddens, ques_hiddens, self.context_mask, self.ques_mask)
            blended_represent = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size_encoder*8) ,论文中的G
            self.bidaf_output = blended_represent
            # 再进过一个双向rnn
            modeling_rnn = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            blended_represent = modeling_rnn.build_graph(blended_represent, self.context_mask, "bidaf_modeling")  # 论文中的M

        else:
            attn_layer = BasicAttention(self.keep_prob)
            _, attn_output = attn_layer.build_graph(context_hiddens, ques_hiddens, self.ques_mask)  # shape=[batch_size,context_len,ques_len]
            blended_represent = tf.concat([context_hiddens, attn_output], axis=2)  # 拼接att和原来rnn encode的输出

        ######################################  Output  ######################################
        if self.FLAGS.bidaf_pointer:
            with vs.variable_scope("StartDist"):
                start_softmax_layer = Bidaf_output_layer(self.FLAGS.context_len, 10 * self.FLAGS.hidden_size_encoder)
                self.logits_start, self.prob_dist_start = start_softmax_layer.build_graph(blended_represent, self.bidaf_output, self.context_mask,
                                                                                          )
            with vs.variable_scope("EndDist"):
                modeling_rnn = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
                M2 = modeling_rnn.build_graph(blended_represent, self.context_mask, "bidaf_modeling")
                end_softmax_layer = Bidaf_output_layer(self.FLAGS.context_len, 10 * self.FLAGS.hidden_size_encoder)
                self.logits_end, self.prob_dist_end = end_softmax_layer.build_graph(M2, self.bidaf_output, self.context_mask,
                                                                                    )

        else:
            blended_reps_final = tf.contrib.layers.fully_connected(blended_represent, num_outputs=self.FLAGS.hidden_size_fully_connected)
            with vs.variable_scope("StartDist"):
                start_softmax_layer = SimpleSoftmaxLayer()
                self.logits_start, self.prob_dist_start = start_softmax_layer.build_graph(blended_reps_final, self.context_mask)
            with vs.variable_scope("EndDist"):
                end_softmax_layer = SimpleSoftmaxLayer()
                self.logits_end, self.prob_dist_end = end_softmax_layer.build_graph(blended_reps_final, self.context_mask)

    def add_loss(self):
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0])
        self.loss_start = tf.reduce_mean(loss_start)
        tf.summary.scalar('loss_start', self.loss_start)  # log to tensorboard

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
        self.loss_end = tf.reduce_mean(loss_end)
        tf.summary.scalar('loss_end', self.loss_end)  # log to tensorboard

        self.loss = self.loss_start + self.loss_end
        tf.summary.scalar('loss', self.loss)  # log to tensorboard

    def padded_char_ids(self, batch, token_ids):
        charids_batch = []
        # char2idx, idx2char, _ = self.create_char_dict()

        for i in range(batch.batch_size):
            charids_line = []
            token_row = token_ids[i]
            for id in token_row:
                token = self.id2word[id]
                char_ids = [self.char2id.get(ch, 1) for ch in list(token)]
                if len(char_ids) > self.FLAGS.word_len:
                    pad_char_ids = char_ids[:self.FLAGS.word_len]
                else:
                    while len(char_ids) < self.FLAGS.word_len:
                        char_ids.append(0)
                    pad_char_ids = char_ids
                charids_line.append(pad_char_ids)
            charids_batch.append(charids_line)

        return charids_batch

    def matrix_multiplication(self, mat, weight):
        # mat [batch_size,seq_len,hidden_size] weight [hidden_size,hidden_size]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert mat_shape[-1] == weight_shape[0]
        mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])
        mul = tf.matmul(mat_reshape, weight)  # multiply只支持2维的矩阵乘法，因此需要先进行reshape
        return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])

    def highway(self, x, size, scope_name, carry_bias=-1.0):
        with tf.variable_scope(scope_name):
            W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='weight_transform')
            b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name='bias_transform')

            W_H = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='transform')
            b_H = tf.Variable(tf.constant(0.1, shape=[size]), name='bias')

        H = tf.nn.relu(self.matrix_multiplication(x, W_H) + b_H)
        # transform gate T
        T = tf.sigmoid(self.matrix_multiplication(x, W_T) + b_T)
        # carry gate C
        C = tf.subtract(1.0, T)
        y = tf.add(tf.multiply(H, T), tf.multiply(x, C))
        return y

    def run_train_iter(self, session, batch, summary_writer):
        feed_dict = {}
        feed_dict[self.context_ids] = batch.context_ids
        feed_dict[self.context_mask] = batch.context_mask
        feed_dict[self.ques_ids] = batch.ques_ids
        feed_dict[self.ques_mask] = batch.ques_mask
        feed_dict[self.ans_span] = batch.ans_span
        feed_dict[self.keep_prob] = 1.0 - self.FLAGS.dropout  # apply dropout
        if self.FLAGS.add_char_embed:
            feed_dict[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            feed_dict[self.char_ids_ques] = self.padded_char_ids(batch, batch.ques_ids)

        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, feed_dict)
        summary_writer.add_summary(summaries, global_step)
        return loss, global_step, param_norm, gradient_norm

    def get_dev_loss(self, session, dev_context_path, dev_ques_path, dev_ans_path):

        print("Calculating dev loss...")
        # logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch = []
        batch_lengths = []

        for batch in get_batch_data(dev_context_path, dev_ques_path, dev_ans_path, self.FLAGS.batch_size, self.word2id, self.FLAGS.context_len,
                                    self.FLAGS.ques_len):
            feed_dict = {}
            feed_dict[self.context_ids] = batch.context_ids
            feed_dict[self.context_mask] = batch.context_mask
            feed_dict[self.ques_ids] = batch.ques_ids
            feed_dict[self.ques_mask] = batch.ques_mask
            feed_dict[self.ans_span] = batch.ans_span
            if self.FLAGS.add_char_embed:
                feed_dict[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
                feed_dict[self.char_ids_ques] = self.padded_char_ids(batch, batch.ques_ids)

            [loss] = session.run([self.loss], feed_dict=feed_dict)
            example_num = batch.batch_size
            loss_per_batch.append(loss * example_num)
            batch_lengths.append(example_num)

        toc = time.time()
        total_num_examples = sum(batch_lengths)
        print("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc - tic))

        dev_loss = sum(loss_per_batch) / float(total_num_examples)
        return dev_loss

    def get_batch_f1_em(self, session, batch):
        # 一个batch数据的预测位置
        pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)  # narray类型
        pred_ans_start = pred_start_pos.tolist()
        pred_ans_end = pred_end_pos.tolist()
        f1_total = 0.0
        em_total = 0.0

        for i, (ans_start, ans_end, context_tokens, ans_tokens) in enumerate(
                zip(pred_ans_start, pred_ans_end, batch.context_tokens, batch.ans_tokens)):
            true_ans = ' '.join(ans_tokens)
            pred_ans_tokens = context_tokens[ans_start:ans_end + 1]
            pred_ans = ' '.join(pred_ans_tokens)
            f1 = f1_score(pred_ans, true_ans)
            em = em_score(pred_ans, true_ans)
            f1_total += f1
            em_total += em

        return f1_total / batch.batch_size, em_total / batch.batch_size

    def get_start_end_pos(self, session, batch):
        feed_dict = {}
        feed_dict[self.context_ids] = batch.context_ids
        feed_dict[self.context_mask] = batch.context_mask
        feed_dict[self.ques_ids] = batch.ques_ids
        feed_dict[self.ques_mask] = batch.ques_mask
        feed_dict[self.ans_span] = batch.ans_span

        if self.FLAGS.add_char_embed:
            feed_dict[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            feed_dict[self.char_ids_ques] = self.padded_char_ids(batch, batch.ques_ids)

        output_feed = [self.prob_dist_start, self.prob_dist_end]
        prob_dist_start, prob_dist_end = session.run(output_feed, feed_dict)

        # 获取开始位置、结束位置的预测
        start_pos = np.argmax(prob_dist_start, axis=1)
        end_pos = np.argmax(prob_dist_end, axis=1)

        return start_pos, end_pos

    def train(self, session, train_context_path, train_ques_path, train_ans_path, dev_context_path, dev_ques_path, dev_ans_path):

        # tensorboard
        summary_dir = self.FLAGS.summary_dir
        summary_writer = tf.summary.FileWriter(summary_dir, session.graph)

        # saver
        saver = tf.train.Saver(max_to_keep=self.FLAGS.keep)
        best_model_saver = tf.train.Saver(max_to_keep=1)
        ckpt = os.path.join(self.FLAGS.ckpt_path, "qa.ckpt")
        best_model_ckpt = os.path.join(self.FLAGS.best_model_ckpt_path, "qa_best.ckpt")

        # F1 EM
        best_F1 = None
        best_EM = None

        epoch = 0
        while self.FLAGS.epochs == 0 or epoch < self.FLAGS.epochs:
            epoch += 1
            print("The {} training epoch".format(epoch))

            for batch in get_batch_data(train_context_path, train_ques_path, train_ans_path, self.FLAGS.batch_size, self.word2id,
                                        self.FLAGS.context_len, self.FLAGS.ques_len):
                loss, global_step, param_norm, gradient_norm = self.run_train_iter(session, batch, summary_writer)

                if global_step % self.FLAGS.print_every == 0:
                    print('epoch %d, global_step %d, loss %.5f,  grad norm %.5f, param norm %.5f' % (
                        epoch, global_step, loss, gradient_norm, param_norm))

                if global_step % self.FLAGS.save_every == 0:
                    # 保存模型
                    print("Saving to {}...".format(ckpt))
                    saver.save(session, ckpt, global_step=global_step)

                if global_step % self.FLAGS.eval_every == 0:

                    # 验证集上的loss
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_ques_path, dev_ans_path)
                    self.add_summary(summary_writer, dev_loss, "dev_loss", global_step)

                    # train F1 EM
                    train_f1, train_em = self.get_batch_f1_em(session, batch)
                    self.add_summary(summary_writer, train_f1, 'train/F1', global_step)
                    self.add_summary(summary_writer, train_em, 'train/EM', global_step)
                    print("train: f1:{},em:{}".format(train_f1, train_em))

                    # dev F1 EM
                    dev_f1 = 0.0
                    dev_em = 0.0
                    i = 0
                    for dev_batch in get_batch_data(dev_context_path, dev_ques_path, dev_ans_path, self.FLAGS.batch_size, self.word2id,
                                                    self.FLAGS.context_len, self.FLAGS.ques_len):
                        f1, em = self.get_batch_f1_em(session, dev_batch)
                        dev_f1 += f1
                        dev_em += em
                        i += 1
                    dev_f1 /= i
                    dev_em /= i
                    self.add_summary(summary_writer, dev_f1, 'dev/F1', global_step)
                    self.add_summary(summary_writer, dev_em, 'dev/EM', global_step)
                    print("dev: f1:{},em:{}".format(dev_f1, dev_em))

                    # 更新最好的模型
                    if best_F1 is None or best_F1 < dev_f1:
                        print("New Better Model！ Saving...")
                        best_model_saver.save(session, best_model_ckpt, global_step=global_step)
                        best_F1 = dev_f1

    def add_summary(self, summary_writer, value, tag, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)
