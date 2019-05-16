# coding=utf-8
import numpy as np
import tensorflow as tf
import os
import time
from data_batcher import Batch
from modules import *
from data_batcher import get_batch_data, pad
from evaluate import f1_score, em_score, print_test_score
from tensorflow.python.ops import variable_scope as vs
from vocab import get_char2id, UNK_ID, PAD_ID, get_tag2id, get_ner2id
from preprocess import tokenize_pos_ner, c2q_match
import logging

logger = logging.getLogger("QA-Model")


class Model(object):
    def __init__(self, FLAGS, embed_matrix, word2id, id2word):
        #### 日志相关 ####
        logging.basicConfig(level=FLAGS.log_level)
        file_handler = logging.FileHandler(os.path.join(FLAGS.summary_dir, "log.txt"))
        logger.addHandler(file_handler)

        self.FLAGS = FLAGS
        self.word2id = word2id
        self.id2word = id2word
        self.tag2id = get_tag2id()
        self.ner2id = get_ner2id()

        # char的种类个数
        char2id, id2char, num_chars = get_char2id()
        self.char2id = char2id
        self.id2char = id2char
        self.num_chars = num_chars

        # 构建图
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.add_placeholder()
            self.add_embedding(embed_matrix)
            self.build_graph()
            self.loss, self.loss_start, self.loss_end = self.add_loss()
            # self.train_op = self.create_optimizer()
            # self.summaries = self.create_summaries()
        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        # opt = tf.contrib.opt.AdaMaxOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
        self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=200, decay_rate=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)  # you can try other optimizers
        self.train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def add_placeholder(self):
        with tf.name_scope("placeholders"):
            self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
            self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
            self.context_pos = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
            self.context_ner = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
            self.context_features = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, 4])
            self.ques_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len])
            self.ques_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len])
            self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

            # 默认的值keep prob
            self.keep_prob = tf.placeholder_with_default(1.0, shape=())

            # char cnn 的 placeholder
            if self.FLAGS.add_char_embed:
                self.char_ids_context = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])
                self.char_ids_ques = tf.placeholder(tf.int32, shape=[None, self.FLAGS.ques_len, self.FLAGS.word_len])

    def add_embedding(self, embed_matrix):
        with vs.variable_scope("embedding"):
            # word embedding
            embedding_matrix = tf.constant(embed_matrix, dtype=tf.float32, name="word_embs")
            context_emb = tf.nn.embedding_lookup(embedding_matrix, self.context_ids)
            self.context_glove_emb = tf.nn.dropout(context_emb, keep_prob=self.keep_prob)
            ques_emb = tf.nn.embedding_lookup(embedding_matrix, self.ques_ids)
            self.ques_emb = tf.nn.dropout(ques_emb, keep_prob=self.keep_prob)
            self.ques_glove_emb = self.ques_emb

            # pos embedding
            pos_embedding_matrix = tf.get_variable(name="pos_embs", shape=[self.FLAGS.pos_nums, self.FLAGS.pos_embedding_size],
                                                   initializer=tf.truncated_normal_initializer(stddev=1))
            context_pos_emb = tf.nn.embedding_lookup(pos_embedding_matrix, self.context_pos)
            context_pos_emb = tf.nn.dropout(context_pos_emb, keep_prob=self.keep_prob)
            logger.debug("context_pos_emb :{}".format(context_pos_emb.shape))

            # ner embedding
            ner_embedding_matrix = tf.get_variable(name="ner_embs", shape=[self.FLAGS.ner_nums, self.FLAGS.ner_embedding_size],
                                                   initializer=tf.truncated_normal_initializer(stddev=1))
            context_ner_emb = tf.nn.embedding_lookup(ner_embedding_matrix, self.context_ner)
            context_ner_emb = tf.nn.dropout(context_ner_emb, keep_prob=self.keep_prob)
            logger.debug("context_ner_emb :{}".format(context_ner_emb.shape))

            # 对齐特征
            aligned_emb = self.align_question_embedding(self.context_glove_emb, self.ques_emb, self.ques_mask)
            logger.debug("aligned_emb :{}".format(aligned_emb.shape))

            # 将所有特征拼接起来
            self.context_emb = tf.concat(
                [self.context_glove_emb, context_pos_emb, context_ner_emb, aligned_emb, tf.cast(self.context_features, tf.float32)], -1)
            logger.debug("after embed context_emb :{}".format(self.context_emb.shape))

    def align_question_embedding(self, context, ques, ques_mask):
        """ Compute Aligned question embedding.

        Args:
            p: context tensor, shape [batch_size, context_len, emb_size]
            q: question tensor, shape [batch_size, question_len, emb_size]

        Return:
            tensor of shape [batch_size,context_len , hidden_size]
        """
        return SeqAttnMatch(context, ques, ques_mask)  # [batch,context_len,h]

    def add_char_embedding(self, config):

        def conv1d(inputs, output_size, width, stride, scope_name):
            length = inputs.get_shape()[-1]  # char embedding_size
            inputs = tf.expand_dims(inputs, axis=3)  # 增加channel 维度
            with vs.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                filter = tf.get_variable("filter", [width, length, 1, output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))

            convolved = tf.nn.conv2d(inputs, filter=filter, strides=[1, stride, 1, 1], padding='VALID')
            logger.debug("conv2d:{}".format(convolved.shape))
            result = tf.squeeze(convolved, axis=2)  # 在squeeze之前：[batch_size, word_len-width, 1, output_size]
            logger.debug("conv2d result:{}".format(result.shape))
            return result

        # 可训练的矩阵，dtype默认为float32
        char_embed_matrix = tf.Variable(tf.random_uniform([self.num_chars, config.char_embedding_size], -1, 1))

        self.context_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_context, shape=(-1, config.word_len)))
        self.context_char_emb = tf.reshape(self.context_char_emb, shape=(-1, config.word_len, config.char_embedding_size))
        logger.debug("context_char_emb :{}".format(self.context_char_emb.shape))
        # shape = batch_size * context_len, word_max_len, char_embedding_size

        self.ques_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_ques, shape=(-1, config.word_len)))
        self.ques_char_emb = tf.reshape(self.ques_char_emb, shape=(-1, config.word_len, config.char_embedding_size))
        logger.debug("ques_char_emb :{}".format(self.context_char_emb.shape))

        # char CNN
        # ############# context ##############
        self.context_cnn_out = conv1d(inputs=self.context_char_emb, output_size=config.char_out_size, width=config.window_width, stride=1,
                                      scope_name='char-cnn')
        self.context_cnn_out = tf.nn.dropout(self.context_cnn_out, self.keep_prob)
        # shape= [batch*context_len,word_len-witdh,char_out_size]
        logger.debug("Shape context embs after conv:{}".format(self.context_cnn_out.shape))

        self.context_cnn_out = tf.reduce_sum(self.context_cnn_out, axis=1)
        self.context_cnn_out = tf.reshape(self.context_cnn_out, shape=(
            -1, config.context_len, config.char_out_size))
        # shape= [batch  context_len, char_out_size]
        logger.debug("Shape context embs after pooling:{}".format(self.context_cnn_out.shape))

        # ############# question #############
        self.ques_cnn_out = conv1d(inputs=self.ques_char_emb, output_size=config.char_out_size, width=config.window_width, stride=1,
                                   scope_name='char-cnn')
        self.ques_cnn_out = tf.nn.dropout(self.ques_cnn_out, self.keep_prob)
        logger.debug("Shape ques embs after conv:{}".format(self.ques_cnn_out.shape))

        self.ques_cnn_out = tf.reduce_sum(self.ques_cnn_out, axis=1)
        self.ques_cnn_out = tf.reshape(self.ques_cnn_out, shape=(-1, config.ques_len, config.char_out_size))
        logger.debug("Shape ques embs after pooling:{}".format(self.ques_cnn_out.shape))

        return self.context_cnn_out, self.ques_cnn_out

    def build_graph(self):

        ######################################  Char Embedding  ######################################
        if self.FLAGS.add_char_embed:
            self.context_char_out, self.ques_char_out = self.add_char_embedding(self.FLAGS)
            # 将word embed和 char embed 拼接
            self.context_emb = tf.concat([self.context_emb, self.context_char_out], axis=2)
            logger.debug("Shape - concatenated context embs:{}".format(self.context_emb.shape))
            self.ques_emb = tf.concat([self.ques_emb, self.ques_char_out], axis=2)
            logger.debug("Shape - concatenated qn embs:{}".format(self.ques_emb.shape))

        ######################################  FFN  ######################################
        # 因为文章加了额外的特征，因此与问题的表示维度不同，为了一致，再引入2层的 FFN 进行变换，得到一样维度的表示
        self.context_emb = FFN(self.context_emb, [self.FLAGS.pwnn_hidden_size, self.FLAGS.pwnn_hidden_size], scope="context_ffn")
        self.ques_emb = FFN(self.ques_emb, [self.FLAGS.pwnn_hidden_size, self.FLAGS.pwnn_hidden_size], scope="ques_ffn")

        ######################################  Highway   ######################################
        if self.FLAGS.add_highway_layer:
            emb_size = self.context_emb.get_shape().as_list()[-1]
            for _ in range(2):
                self.context_emb = self.highway(self.context_emb, emb_size, scope_name='highway')
                self.ques_emb = self.highway(self.ques_emb, emb_size, scope_name='highway')

        ######################################  RNNEncoder  ######################################

        # 共用同一个编码器
        # [batch,contex_len,4d]
        self.context_l = create_rnn_graph(1, self.FLAGS.hidden_size, self.context_emb, self.context_mask, "context_low_level")
        self.context_h = create_rnn_graph(1, self.FLAGS.hidden_size, self.context_l, self.context_mask, "context_high_level")

        self.ques_l = create_rnn_graph(1, self.FLAGS.hidden_size, self.ques_emb, self.ques_mask, "ques_low_level")
        self.ques_h = create_rnn_graph(1, self.FLAGS.hidden_size, self.ques_l, self.ques_mask, "ques_high_level")
        self.ques_u = create_rnn_graph(1, self.FLAGS.hidden_size, tf.concat([self.ques_l, self.ques_h], axis=2), self.ques_mask, "ques_understand")

        # 将问题 summary 为向量
        w = SelfAttn(self.ques_u, self.ques_mask)  # [batch,len]
        w = tf.reshape(w, [-1, 1, self.FLAGS.ques_len])
        encoded_question = tf.einsum('ijk,ikq->ijq', w, self.ques_u)  # b x 1 x 2*hidden_size
        encoded_question = tf.reshape(encoded_question, [-1, 2 * self.FLAGS.hidden_size])  # [batch, 2*hidden_size]

        ######################################  Attention  ######################################
        # last_dim=context_hiddens.shape().as_list()[-1]
        if self.FLAGS.Chen:
            with vs.variable_scope("attention", reuse=False):
                # 文章的 attention
                atten_context_hiddens = SeqAttnMatch(self.context_h, self.ques_u)
                context_hiddens = tf.concat([self.context_h, atten_context_hiddens], axis=2)

            with vs.variable_scope("self_attention", reuse=False):
                # self attention
                self_atten_context_hiddens = SeqAttnMatch(context_hiddens, context_hiddens)
                context_hiddens = tf.concat([context_hiddens, self_atten_context_hiddens], axis=2)

            with vs.variable_scope("model"):
                context_hiddens = create_rnn_graph(1, self.FLAGS.hidden_size, context_hiddens, self.context_mask, "model")  # [batch,contex_len,2d]

        elif self.FLAGS.fusion:
            # 历史信息拼接起来
            How_q = tf.concat([self.ques_glove_emb, self.ques_l, self.ques_h], axis=2)
            How_c = tf.concat([self.context_glove_emb, self.context_l, self.context_h], axis=2)
            logger.debug("HoW_c 的 shape：{}".format(How_c.shape))
            logger.debug("How_q 的 shape：{}".format(How_q.shape))
            with vs.variable_scope("low_level_fusion"):
                self.attended_context_l = fusion_attention(How_c, How_q, self.ques_l, self.ques_mask, self.FLAGS.fusion_att_hidden_size)
            with vs.variable_scope("high_level_fusion"):
                self.attended_context_h = fusion_attention(How_c, How_q, self.ques_h, self.ques_mask, self.FLAGS.fusion_att_hidden_size)
            with vs.variable_scope("understand_level_fusion"):
                self.attended_context_understand = fusion_attention(How_c, How_q, self.ques_u, self.ques_mask, self.FLAGS.fusion_att_hidden_size)

            How_c_ = tf.concat([self.context_l, self.context_h, self.attended_context_l, self.attended_context_h, self.attended_context_understand],
                               axis=2)
            self.context_v = create_rnn_graph(1, self.FLAGS.hidden_size, How_c_, self.context_mask, "context_1")

            How_c_full = tf.concat(
                [self.context_glove_emb, self.context_l, self.context_h, self.attended_context_l, self.attended_context_h,
                 self.attended_context_understand,
                 self.context_v], axis=2)

            # self attention
            self.self_attended_context = fusion_attention(How_c_full, How_c_full, self.context_v, self.context_mask,
                                                          self.FLAGS.fusion_att_hidden_size)
            self.context_uu = create_rnn_graph(1, self.FLAGS.hidden_size, tf.concat([self.context_v, self.self_attended_context], 2),
                                               self.context_mask, "context_2")


        elif self.FLAGS.bidaf_attention:

            # attention 层
            attn_layer = Bidaf(self.keep_prob, self.FLAGS.hidden_size_encoder * 2)
            attn_output = attn_layer.build_graph(self.context_h, self.ques_u, self.context_mask, self.ques_mask)
            blended_represent = tf.concat([self.context_h, attn_output], axis=2)  # (batch_size, context_len, hidden_size_encoder*8) ,论文中的G
            self.bidaf_output = blended_represent

            # 再进过一个双向rnn
            modeling_rnn = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            blended_represent = modeling_rnn.build_graph(blended_represent, self.context_mask, "bidaf_modeling")  # 论文中的M

        else:
            attn_layer = BasicAttention(self.keep_prob)
            _, attn_output = attn_layer.build_graph(self.context_h, self.ques_u, self.ques_mask)  # shape=[batch_size,context_len,ques_len]
            blended_represent = tf.concat([self.context_h, attn_output], axis=2)  # 拼接att和原来rnn encode的输出

        ######################################  Output  ######################################

        if self.FLAGS.Chen:
            with vs.variable_scope("start_dist"):
                logits_start = bilinear_sequnce_attention(context_hiddens, encoded_question)  # [batch,context_len]
                self.logits_start, self.prob_dist_start = masked_softmax(logits_start, self.context_mask, 1)
            with vs.variable_scope("end_dist"):
                logits_end = bilinear_sequnce_attention(context_hiddens, encoded_question)
                self.logits_end, self.prob_dist_end = masked_softmax(logits_end, self.context_mask, 1)

        elif self.FLAGS.fusion:
            with vs.variable_scope("start_dist"):
                logits_start = bilinear_sequnce_attention(self.context_uu, encoded_question)  # [batch,context_len]
                self.logits_start, self.prob_dist_start = masked_softmax(logits_start, self.context_mask, 1)
            with vs.variable_scope("end_dist"):
                self.with_start_context = tf.expand_dims(self.prob_dist_start, axis=2) * self.context_uu
                self.v_q = tf.reduce_sum(self.with_start_context, axis=1)
                # TODO gru
                logits_end = bilinear_sequnce_attention(self.context_uu, self.v_q)
                self.logits_end, self.prob_dist_end = masked_softmax(logits_end, self.context_mask, 1)

        elif self.FLAGS.bidaf_pointer:
            with vs.variable_scope("StartDist"):
                start_softmax_layer = Bidaf_output_layer(self.FLAGS.context_len, 10 * self.FLAGS.hidden_size_encoder)
                self.logits_start, self.prob_dist_start = start_softmax_layer.build_graph(blended_represent, self.bidaf_output, self.context_mask)
            with vs.variable_scope("EndDist"):
                modeling_rnn = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
                M2 = modeling_rnn.build_graph(blended_represent, self.context_mask, "bidaf_modeling")
                end_softmax_layer = Bidaf_output_layer(self.FLAGS.context_len, 10 * self.FLAGS.hidden_size_encoder)
                self.logits_end, self.prob_dist_end = end_softmax_layer.build_graph(M2, self.bidaf_output, self.context_mask)

        elif self.FLAGS.answer_pointer:
            hidden_size_attn = 2 * self.FLAGS.hidden_size_modeling
            pointer = Answer_Pointer(self.keep_prob, self.FLAGS.hidden_size_encoder, self.FLAGS.ques_len, hidden_size_attn)
            p, logits = pointer.build_graph_answer_pointer(blended_represent, self.ques_u, self.ques_mask, self.context_mask, self.FLAGS.context_len)

            self.logits_start = logits[0]
            self.prob_dist_start = p[0]
            self.logits_end = logits[1]
            self.prob_dist_end = p[1]

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
        loss_start = tf.reduce_mean(loss_start)

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
        loss_end = tf.reduce_mean(loss_end)

        loss = loss_start + loss_end
        return loss, loss_start, loss_end

    # def create_optimizer(self):
    #     # 梯度剪切
    #     params = tf.trainable_variables()
    #     gradients = tf.gradients(self.loss, params)
    #     self.gradient_norm = tf.global_norm(gradients)
    #     clipped_gradients, self.global_norm = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
    #     self.param_norm = tf.global_norm(params)
    #
    #     # train op
    #     opt = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate)
    #     train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    #     return train_op
    #
    # def create_summaries(self):
    #     tf.summary.scalar('loss', self.loss)
    #     return tf.summary.merge_all()

    def get_feed_dict(self, batch, keep_prob=None):
        '''
        创建 session 运行的 feed——dict
        :param batch: batch数据
        :param keep_prob:  keep_prob
        :return: feed_dict
        '''
        feed_dict = {
            self.context_ids: batch.context_ids,
            self.context_pos: batch.context_poses,
            self.context_ner: batch.context_ners,
            self.context_features: batch.context_features,
            self.context_mask: batch.context_mask,
            self.ques_ids: batch.ques_ids,
            self.ques_mask: batch.ques_mask,
            self.ans_span: batch.ans_span,
            self.keep_prob: keep_prob if keep_prob is not None else (1.0 - self.FLAGS.dropout)  # apply dropout
        }
        if self.FLAGS.add_char_embed:
            feed_dict[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            feed_dict[self.char_ids_ques] = self.padded_char_ids(batch, batch.ques_ids)
        return feed_dict

    def train_step(self, session, batch):
        '''
        进行模型一个 batch 数据的训练
        :param session: tf session
        :param batch:  一个 batch 对象
        :return: loss, global_step, param_norm, gradient_norm
        '''
        feed_dict = self.get_feed_dict(batch)
        output_feed = [self.train_op, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        [_, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, feed_dict=feed_dict)
        return loss, global_step, param_norm, gradient_norm

    def pack_model_output(self, batch, pred_start_pos, pred_end_pos, pred_start_probs, pred_end_probs):
        '''
        将模型的预测输出封装为指定的字典形式
        :param batch: 传给模型的 batch 数据
        :param pred_start_pos: 预测的开始位置 batch*1 形式
        :param pred_end_pos: 预测的结束位置 batch*1 形式
        :param pred_start_probs: 文章每个单词 softmax 后的作为开始位置概率，batch*context_len
        :param pred_end_probs:   文章每个单词 softmax 后的作为结束位置概率，batch*context_len
        :return:
        '''
        answer_starts = pred_start_pos.tolist()
        answer_ends = pred_end_pos.tolist()
        start_probs = pred_start_probs.tolist()
        end_probs = pred_end_probs.tolist()

        uuid2ans = dict()
        for i, (pred_start, pred_end) in enumerate(zip(answer_starts, answer_ends)):
            context_tokens = batch.context_tokens[i]
            # 确保预测的范围在文章长度内
            assert pred_start in range(len(context_tokens))
            assert pred_end in range(len(context_tokens))

            # 截取预测的答案
            uuid = batch.uuids[i]
            result = dict()
            result['tokens'] = context_tokens
            result['predict_answer'] = ' '.join(context_tokens[pred_start: pred_end + 1])
            result['start_probs'] = np.round(start_probs[i], 2).tolist()
            result['end_probs'] = np.round(end_probs[i], 2).tolist()
            uuid2ans[uuid] = result
        return uuid2ans

    def validate(self, session):
        '''
        进行整个验证集测试，获取 loss，f1，em 的情况
        :param session:
        :return:
        '''
        logger.info("Calculating dev loss\F1\EM...")
        tic = time.time()

        batch_lengths = []
        batch_loss = []
        batch_f1 = []
        batch_em = []

        for batch in get_batch_data(self.FLAGS, "dev", self.word2id):
            f1_avg, em_avg, loss_avg = self.get_batch_f1_em(session, batch)
            logger.debug("f1:{:.2f},em:{:.2f}".format(f1_avg,em_avg))
            example_num = batch.batch_size
            batch_loss.append(loss_avg * example_num)
            batch_f1.append(f1_avg * example_num)
            batch_em.append(em_avg * example_num)
            batch_lengths.append(example_num)

        total_num_examples = float(sum(batch_lengths))
        dev_loss = sum(batch_loss) / total_num_examples
        dev_f1 = sum(batch_f1) / total_num_examples
        dev_em = sum(batch_em) / total_num_examples

        toc = time.time()
        print("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc - tic))
        return dev_loss, dev_f1, dev_em

    def test(self, session):
        '''
        使用dev 数据作为测试数据，进行模型的评价
        :param session:
        :return:
        '''

        logger.info("Calculating test F1\EM...")
        tic = time.time()
        uuid2ans = {}  # uuid->预测答案 的字典

        for batch in get_batch_data(self.FLAGS, "dev", self.word2id):
            start_poses, end_poses, loss_avg, start_probs, end_probs = self.get_answer_pos(session, batch)  # pred_end_pos 是 narray类型
            batch_result = self.pack_model_output(batch, start_poses, end_poses, start_probs, end_probs)
            uuid2ans.update(batch_result)

        toc = time.time()
        logger.info("Computed test F1\EM in %.2f seconds" % (toc - tic))
        return uuid2ans

    def convert_text_as_batch(self, context, question):

        # 转为 unicode
        context = context.decode("utf-8")
        question = question.decode("utf-8")

        # 1.分词，pos，ner，词干，tf
        logger.debug("#"*20+" 开始转换为模型的输入 "+"#"*20)
        t1 = time.time()
        context_tokens, context_poses, context_ners, context_lemmas, context_tf = tokenize_pos_ner(context)
        question_tokens, question_poses, question_ners, question_lemmas, _ = tokenize_pos_ner(question)
        t2 = time.time()
        logger.debug("分词、词性、实体处理: {}s".format(t2-t1))

        # 2.获取文章与问题的3种匹配特征
        question_tokens_lower = [w.lower() for w in question_tokens]
        context_tokens_lower = [w.lower() for w in context_tokens]
        exact_match = c2q_match(context_tokens, set(question_tokens))  # 精确匹配
        lower_match = c2q_match(context_tokens_lower, set(question_tokens_lower))  # 小写匹配
        lemma_match = c2q_match(context_lemmas, set(question_lemmas))  # 提取文章 token 的词干是否出现在问题中
        context_tf_match = [[f1, f2, f3, f4] for f1, f2, f3, f4 in zip(context_tf, exact_match, lower_match, lemma_match)]
        t3 = time.time()
        logger.debug("文章与问题匹配处理: {}s".format(t3 - t2))

        # 3. 转为 id
        context_ids = [self.word2id.get(token.lower(), UNK_ID) for token in context_tokens]
        ques_ids = [self.word2id.get(token.lower(), UNK_ID) for token in question_tokens]
        context_pos_ids = [self.tag2id.get(p, 0) for p in context_poses]
        context_ner_ids = [self.ner2id.get(n, 0) for n in context_ners]
        t4 = time.time()
        logger.debug("token->id 处理: {}s".format(t4 - t3))

        def batchry(item):
            ''' 将一条数据封装为 batch 的形式，其实就是加一层[]'''
            return [item]

        batch_uuids = batchry(1)
        batch_context_ids = batchry(context_ids)
        batch_context_tokens = batchry(context_tokens)
        batch_context_pos_ids = batchry(context_pos_ids)
        batch_context_ner_ids = batchry(context_ner_ids)
        batch_context_features = batchry(context_tf_match)
        batch_ques_ids = batchry(ques_ids)
        batch_ques_tokens = batchry(question_tokens)
        batch_ans_span = batchry([0, 1])
        batch_ans_tokens = batchry([])

        # 进行pad
        context_len = self.FLAGS.context_len
        ques_len = self.FLAGS.ques_len
        batch_context_ids = pad(batch_context_ids, context_len)
        batch_context_pos_ids = pad(batch_context_pos_ids, context_len)
        batch_context_ner_ids = pad(batch_context_ner_ids, context_len)
        batch_ques_ids = pad(batch_ques_ids, ques_len)
        batch_context_features = pad(batch_context_features, context_len, np.array([0, 0, 0, 0]))

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
        t5 = time.time()
        logger.debug("封装为 batch 处理: {}s".format(t5 - t4))
        logger.debug("#"*10+" 完成转换为模型的输入，共耗时：{} ".format(t5-t1)+"#"*10)

        return batch

    def predict_single(self, session, context, question):
        '''
        根据页面传来的单个 文章字符串、问题字符串，利用模型预测答案
        :param session: TensorFlow session 对象
        :param context: 文章 str
        :param question: 问题 str
        :return:
        '''
        # 转为 batch 对象，虽然 batch 中只有一条数据
        batch = self.convert_text_as_batch(context, question)
        start_pos, end_pos, loss, prob_dist_start, prob_dist_end = self.get_answer_pos(session, batch)
        # 将模型的输出封装为 uuid->{页面所需字段}的 json 对象
        uuid2answer = self.pack_model_output(batch, start_pos, end_pos, prob_dist_start, prob_dist_end)
        return uuid2answer

    def get_batch_f1_em(self, session, batch):
        '''
        一个batch数据的f1，em，loss 的平均值
        :param session:
        :param batch:
        :return:
        '''
        pred_start_pos, pred_end_pos, loss_avg, _, _ = self.get_answer_pos(session, batch)  # pred_end_pos 是 narray类型
        pred_ans_start = pred_start_pos.tolist()
        pred_ans_end = pred_end_pos.tolist()
        f1 = 0.0
        em = 0.0
        total = float(batch.batch_size)

        for i, (ans_start, ans_end, ans_tokens) in enumerate(zip(pred_ans_start, pred_ans_end, batch.ans_tokens)):
            true_ans = ' '.join(ans_tokens)
            pred_ans = ' '.join(batch.context_tokens[i][ans_start:ans_end + 1])  # 不放在 zip 中省内存
            f1 += f1_score(pred_ans, true_ans)
            em += em_score(pred_ans, true_ans)

        f1_avg = f1 / total
        em_avg = em / total
        return f1_avg, em_avg, loss_avg

    def get_answer_pos(self, session, batch):
        '''
        获取 batch 数据预测的开始于结束位置
        :param session:
        :param batch:  batch 数据
        :return: start_pos, end_pos [batch_size]
        '''
        t1 = time.time()
        feed_dict = self.get_feed_dict(batch, keep_prob=1)
        output_feed = [self.loss, self.prob_dist_start, self.prob_dist_end]  # 获取 开始位置的概率、结束位置的概率
        loss, prob_dist_start, prob_dist_end = session.run(output_feed, feed_dict)
        t2 = time.time()
        logger.debug("模型预测运行时间: {}s".format(t2 - t1))

        # 获取开始位置、结束位置的预测
        if self.FLAGS.smart_span:
            t3 = time.time()
            curr_batch_size = batch.batch_size
            start_pos = np.empty(shape=(curr_batch_size), dtype=int)
            end_pos = np.empty(shape=(curr_batch_size), dtype=int)
            maxprob = np.empty(shape=(curr_batch_size), dtype=float)

            for j in range(curr_batch_size):  # for each row
                ## Take argmax of start and end dist in a window such that  i <= j <= i + 15
                maxprod = 0
                chosen_start = 0
                chosen_end = 0
                for i in range(self.FLAGS.context_len - 16):
                    end_dist_subset = prob_dist_end[j, i:i + 16]
                    end_prob_max = np.amax(end_dist_subset)
                    end_idx = np.argmax(end_dist_subset)
                    start_prob = prob_dist_start[j, i]
                    prod = end_prob_max * start_prob
                    # print("Prod: ", prod)

                    # print("Shape end, start:", end_prob_max.shape, start_prob.shape)

                    if prod > maxprod:
                        maxprod = prod
                        chosen_start = i
                        chosen_end = chosen_start + end_idx

                start_pos[j] = chosen_start
                # end_idx = np.argmax(end_dist[j:chosen_start:chosen_start+16])
                # print("Chosen end", chosen_start+end_idx)
                end_pos[j] = chosen_end
                maxprob[j] = round(maxprod, 4)

                ## add sanity check
                delta = end_pos[j] - start_pos[j]
                if delta < 0 or delta > 16:
                    logger.error("Error! Please look ,smart span part")
            t4 = time.time()
            logger.debug("smart_span 处理时间: {}s".format(t4 - t3))
        else:
            start_pos = np.argmax(prob_dist_start, axis=1)
            end_pos = np.argmax(prob_dist_end, axis=1)

        return start_pos, end_pos, loss, prob_dist_start, prob_dist_end

    def train(self, session):
        config = self.FLAGS

        # tensorboard
        train_summary_writer = tf.summary.FileWriter(config.summary_dir + "/train/", session.graph)
        valid_summary_writer = tf.summary.FileWriter(config.summary_dir + "/dev/", session.graph)

        # saver
        saver = tf.train.Saver(max_to_keep=config.keep)
        best_model_saver = tf.train.Saver(max_to_keep=1)
        ckpt = os.path.join(config.ckpt_path, "qa.ckpt")
        best_model_ckpt = os.path.join(config.best_model_ckpt_path, "qa_best.ckpt")

        # F1 EM
        best_F1 = None
        best_EM = None

        loss_save = 100.0
        patience = 0
        lr = config.learning_rate
        epoch = 0

        session.run(tf.assign(self.lr, tf.constant(lr, dtype=tf.float32)))
        while config.epochs == 0 or epoch < config.epochs:
            epoch += 1
            logger.info("#" * 50)
            logger.info("The {} training epoch".format(epoch))

            for batch in get_batch_data(config, "train", self.word2id):
                loss, global_step, param_norm, gradient_norm = self.train_step(session, batch)
                # 打印
                if global_step % config.print_every == 0:
                    logger.info('epoch %d, global_step %d, loss %.5f,  grad norm %.5f,  param norm %.5f' % (
                        epoch, global_step, loss, gradient_norm, param_norm))

                # 保存模型
                if global_step % config.save_every == 0:
                    logger.info("Saving to {}...".format(ckpt))
                    saver.save(session, ckpt, global_step=global_step)

                # 验证
                if global_step % config.eval_every == 0:
                    # 训练集 F1 EM
                    train_f1, train_em, train_loss = self.get_batch_f1_em(session, batch)
                    self.add_summary(train_summary_writer, train_f1, 'train/F1', global_step)
                    self.add_summary(train_summary_writer, train_em, 'train/EM', global_step)
                    logger.info("train: f1:{},em:{}".format(train_f1, train_em))

                    # 验证集 loss F1 EM
                    dev_loss, dev_f1, dev_em = self.validate(session)
                    self.add_summary(valid_summary_writer, dev_loss, "dev/loss", global_step)
                    self.add_summary(valid_summary_writer, dev_f1, 'dev/F1', global_step)
                    self.add_summary(valid_summary_writer, dev_em, 'dev/EM', global_step)
                    logger.info("dev: loss:{}, f1:{},em:{}".format(dev_loss, dev_f1, dev_em))
                    # print("lr:{}".format(self.lr))
                    # uuid2ans = self.test(session)
                    # with codecs.open(config.predict_answer_file, 'w', encoding='utf-8') as f:
                    #     ans = unicode(json.dumps(uuid2ans, ensure_ascii=False))
                    #     f.write(ans)
                    # # 3.评价
                    # result = print_test_score()
                    # dev_f1, dev_em = result['f1'],result['em']
                    # self.add_summary(valid_summary_writer, result['f1'], 'dev/F1', global_step)
                    # self.add_summary(valid_summary_writer, result['em'], 'dev/EM', global_step)

                    # 更新最好的模型
                    if best_F1 is None or best_F1 < dev_f1:
                        logger.info("New Better Model！Saving to {}...".format(best_model_ckpt))
                        best_model_saver.save(session, best_model_ckpt, global_step=global_step)
                        best_F1 = dev_f1

                    if dev_loss < loss_save:
                        loss_save = dev_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= config.patience:
                        lr /= 2.0
                        loss_save = dev_loss
                        patience = 0
                    session.run(tf.assign(self.lr, tf.constant(lr, dtype=tf.float32)))

        train_summary_writer.close()
        valid_summary_writer.close()

    def add_summary(self, summary_writer, value, tag, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)

    def padded_char_ids(self, batch, batch_token_ids):
        charids_batch = []
        # char2idx, idx2char, _ = self.create_char_dict()

        for i in range(batch.batch_size):
            charids_line = []
            token_ids = batch_token_ids[i]
            # 一句话的 id
            for id in token_ids:
                token = self.id2word[id]
                char_ids = [self.char2id.get(ch, 1) for ch in list(token)]
                if len(char_ids) > self.FLAGS.word_len:
                    pad_char_ids = char_ids[:self.FLAGS.word_len]
                else:
                    char_ids.extend([0] * (self.FLAGS.word_len - len(char_ids)))
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
        mul = tf.matmul(mat_reshape, weight)  # matmul的矩阵乘法，因此需要先进行reshape
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
