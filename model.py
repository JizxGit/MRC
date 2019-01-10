# coding=utf-8
import numpy as np
import tensorflow as tf
import os
import time
from modules import *
from data_batcher import get_batch_data
from evaluate import f1_score, em_score
from tensorflow.python.ops import variable_scope as vs
from vocab import get_char2id


class Model(object):
    def __init__(self, FLAGS, embed_matrix, word2id, id2word):
        self.FLAGS = FLAGS
        self.word2id = word2id
        self.id2word = id2word

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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # opt = tf.contrib.opt.AdaMaxOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
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
            self.context_emb = tf.nn.dropout(context_emb, keep_prob=self.keep_prob)
            ques_emb = tf.nn.embedding_lookup(embedding_matrix, self.ques_ids)
            self.ques_emb = tf.nn.dropout(ques_emb, keep_prob=self.keep_prob)

            # pos embedding
            pos_embedding_matrix = tf.get_variable(name="pos_embs", shape=[self.FLAGS.pos_nums, self.FLAGS.pos_embedding_size],
                                                   initializer=tf.truncated_normal_initializer(stddev=1), trainable=False)
            context_pos_emb = tf.nn.embedding_lookup(pos_embedding_matrix, self.context_pos)
            context_pos_emb = tf.nn.dropout(context_pos_emb, keep_prob=self.keep_prob)
            print("context_pos_emb :", context_pos_emb.shape)
            # ner embedding
            ner_embedding_matrix = tf.get_variable(name="ner_embs", shape=[self.FLAGS.ner_nums, self.FLAGS.ner_embedding_size],
                                                   initializer=tf.truncated_normal_initializer(stddev=1), trainable=False)
            context_ner_emb = tf.nn.embedding_lookup(ner_embedding_matrix, self.context_ner)
            context_ner_emb = tf.nn.dropout(context_ner_emb, keep_prob=self.keep_prob)
            print("context_ner_emb :", context_ner_emb.shape)

            # 对齐特征
            aligned_emb = self.align_question_embedding(self.context_emb, self.ques_emb)
            print("aligned_emb :", aligned_emb.shape)
            # 将所有特征拼接起来

            self.context_emb = tf.concat([self.context_emb, context_pos_emb, context_ner_emb, aligned_emb, tf.cast(self.context_features, tf.float32)], -1)
            # self.context_emb = tf.concat([self.context_emb, context_pos_emb, context_ner_emb, aligned_emb], -1)
            print("after embed context_emb :", self.context_emb.shape)


    def align_question_embedding(self, p, q):
        """ Compute Aligned question embedding.

        Args:
            p: context tensor, shape [batch_size,context_len, emb_size]
            q: question tensor, shape [batch_size,question_len, emb_size]

        Return:
            tensor of shape [batch_size,context_len , hidden_size]
        """
        # TODO 确认返回的 shape
        return SeqAttnMatch(p, q)

    def add_char_embedding(self, config):

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
        char_embed_matrix = tf.Variable(tf.random_uniform([self.num_chars, config.char_embedding_size], -1, 1))

        self.context_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_context, shape=(-1, config.word_len)))
        self.context_char_emb = tf.reshape(self.context_char_emb, shape=(-1, config.word_len, config.char_embedding_size))
        print("context_char_emb :", self.context_char_emb.shape)
        # shape = batch_size * context_len, word_max_len, char_embedding_size

        self.ques_char_emb = tf.nn.embedding_lookup(char_embed_matrix, tf.reshape(self.char_ids_ques, shape=(-1, config.word_len)))
        self.ques_char_emb = tf.reshape(self.ques_char_emb, shape=(-1, config.word_len, config.char_embedding_size))
        print("ques_char_emb :", self.context_char_emb.shape)

        # char CNN
        # ############# context ##############
        self.context_cnn_out = conv1d(inputs=self.context_char_emb, output_size=config.char_out_size, width=config.window_width, stride=1,
                                      scope_name='char-cnn')
        self.context_cnn_out = tf.nn.dropout(self.context_cnn_out, self.keep_prob)
        # shape= [batch*context_len,word_len-witdh,char_out_size]
        print("Shape context embs after conv", self.context_cnn_out.shape)

        self.context_cnn_out = tf.reduce_sum(self.context_cnn_out, axis=1)
        self.context_cnn_out = tf.reshape(self.context_cnn_out, shape=(
            -1, config.context_len, config.char_out_size))
        # shape= [batch  context_len, char_out_size]
        print("Shape context embs after pooling", self.context_cnn_out.shape)

        # ############# question #############
        self.ques_cnn_out = conv1d(inputs=self.ques_char_emb, output_size=config.char_out_size, width=config.window_width, stride=1,
                                   scope_name='char-cnn')
        self.ques_cnn_out = tf.nn.dropout(self.ques_cnn_out, self.keep_prob)
        print("Shape ques embs after conv", self.ques_cnn_out.shape)

        self.ques_cnn_out = tf.reduce_sum(self.ques_cnn_out, axis=1)
        self.ques_cnn_out = tf.reshape(self.ques_cnn_out, shape=(-1, config.ques_len, config.char_out_size))
        print("Shape ques embs after pooling", self.ques_cnn_out.shape)

        return self.context_cnn_out, self.ques_cnn_out

    def build_graph(self):

        ######################################  Char Embedding  ######################################
        if self.FLAGS.add_char_embed:
            self.context_char_out, self.ques_char_out = self.add_char_embedding(self.FLAGS)
            # 将word embed和 char embed 拼接
            self.context_emb = tf.concat([self.context_emb, self.context_char_out], axis=2)
            print("Shape - concatenated context embs", self.context_emb.shape)
            self.ques_emb = tf.concat([self.ques_emb, self.ques_char_out], axis=2)
            print("Shape - concatenated qn embs", self.ques_emb.shape)

        ######################################  Highway   ######################################
        if self.FLAGS.add_highway_layer:
            emb_size = self.context_emb.get_shape().as_list()[-1]
            for _ in range(2):
                self.context_emb = self.highway(self.context_emb, emb_size, scope_name='highway')
                self.ques_emb = self.highway(self.ques_emb, emb_size, scope_name='highway')

        ######################################  RNNEncoder  ######################################

        # encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        # context_hiddens = encoder.build_graph(self.context_emb, self.context_mask, scope_name="RNNEncoder")
        # ques_hiddens = encoder.build_graph(self.ques_emb, self.ques_mask, scope_name="RNNEncoder")
        # self.bug = tf.check_numerics(self.context_emb, "NaN")
        with vs.variable_scope("context_encoder"):
            context_hiddens = create_rnn_graph(self.FLAGS.rnn_layer_num, self.FLAGS.hidden_size, self.context_emb, self.context_mask,
                                               "context")  # [batch,contex_len,4d]
        # self.bug = tf.check_numerics(context_hiddens, "NaN")
        with vs.variable_scope("question_encoder"):
            ques_hiddens = create_rnn_graph(self.FLAGS.rnn_layer_num, self.FLAGS.hidden_size, self.ques_emb, self.ques_mask, "question")

        # 将问题 encode 为向量
        weights = SelfAttn(ques_hiddens,self.ques_mask)  # [batch,len]
        weights = tf.reshape(weights, [-1, 1, self.FLAGS.ques_len])
        encoded_question = tf.einsum('ijk,ikq->ijq', weights, ques_hiddens)  # b x 1 x 4*hidden_size
        encoded_question = tf.reshape(encoded_question, [-1, 2 * self.FLAGS.rnn_layer_num * self.FLAGS.hidden_size])  # [batch, 4*hidden_size]

        ######################################  Basic Attention  ######################################
        # last_dim=context_hiddens.shape().as_list()[-1]
        if self.FLAGS.Chen:

            # TODO 文章的 attention 与 self attention

            pass

        elif self.FLAGS.bidaf_attention:

            # attention 层
            attn_layer = Bidaf(self.keep_prob, self.FLAGS.hidden_size_encoder * 2)
            attn_output = attn_layer.build_graph(context_hiddens, ques_hiddens, self.context_mask, self.ques_mask)
            blended_represent = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size_encoder*8) ,论文中的G
            self.bidaf_output = blended_represent
            # 再进过一个双向rnn
            modeling_rnn = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            blended_represent = modeling_rnn.build_graph(blended_represent, self.context_mask, "bidaf_modeling")  # 论文中的M
            # TODO self attention

        else:
            attn_layer = BasicAttention(self.keep_prob)
            _, attn_output = attn_layer.build_graph(context_hiddens, ques_hiddens, self.ques_mask)  # shape=[batch_size,context_len,ques_len]
            blended_represent = tf.concat([context_hiddens, attn_output], axis=2)  # 拼接att和原来rnn encode的输出

        ######################################  Output  ######################################

        if self.FLAGS.Chen:
            with vs.variable_scope("start_dist"):
                logits_start = bilinear_sequnce_attention(context_hiddens, encoded_question)  # [batch,context_len]
                self.logits_start, self.prob_dist_start = masked_softmax(logits_start, self.context_mask, 1)
            with vs.variable_scope("end_dist"):
                logits_end = bilinear_sequnce_attention(context_hiddens, encoded_question)
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
            p, logits = pointer.build_graph_answer_pointer(blended_represent, ques_hiddens, self.ques_mask, self.context_mask, self.FLAGS.context_len)

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
        [_ ,loss, global_step, param_norm, gradient_norm] = session.run(output_feed, feed_dict=feed_dict)
        return loss, global_step, param_norm, gradient_norm

    def validate(self, session):
        '''
        进行整个验证集测试，获取 loss，f1，em 的情况
        :param session:
        :return:
        '''
        print("Calculating dev loss\F1\EM...")
        # logging.info("Calculating dev loss...")
        tic = time.time()

        batch_lengths = []
        batch_loss = []
        batch_f1 = []
        batch_em = []

        for batch in get_batch_data(self.FLAGS, "dev", self.word2id):
            f1_avg, em_avg, loss_avg = self.get_batch_f1_em(session, batch)
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

    def get_batch_f1_em(self, session, batch):
        '''
        一个batch数据的f1，em，loss 的平均值
        :param session:
        :param batch:
        :return:
        '''
        pred_start_pos, pred_end_pos, loss_avg = self.get_answer_pos(session, batch)  # pred_end_pos 是 narray类型
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
        feed_dict = self.get_feed_dict(batch, keep_prob=1)
        output_feed = [self.loss, self.prob_dist_start, self.prob_dist_end]
        loss, prob_dist_start, prob_dist_end = session.run(output_feed, feed_dict)

        # 获取开始位置、结束位置的预测
        start_pos = np.argmax(prob_dist_start, axis=1)
        end_pos = np.argmax(prob_dist_end, axis=1)
        return start_pos, end_pos, loss

    def train(self, session):
        config = self.FLAGS

        # tensorboard
        train_summary_writer = tf.summary.FileWriter(config.summary_dir+"/train/", session.graph)
        valid_summary_writer = tf.summary.FileWriter(config.summary_dir+"/dev/", session.graph)

        # saver
        saver = tf.train.Saver(max_to_keep=config.keep)
        best_model_saver = tf.train.Saver(max_to_keep=1)
        ckpt = os.path.join(config.ckpt_path, "qa.ckpt")
        best_model_ckpt = os.path.join(config.best_model_ckpt_path, "qa_best.ckpt")

        # F1 EM
        best_F1 = None
        best_EM = None

        epoch = 0
        while config.epochs == 0 or epoch < config.epochs:
            epoch += 1
            print("#"*50)
            print("The {} training epoch".format(epoch))

            for batch in get_batch_data(config, "train", self.word2id):
                loss, global_step, param_norm, gradient_norm = self.train_step(session, batch)
                # 打印
                if global_step % config.print_every == 0:
                    print('epoch %d, global_step %d, loss %.5f,  grad norm %.5f,  param norm %.5f' % (
                        epoch, global_step, loss, gradient_norm, param_norm))

                # 保存模型
                if global_step % config.save_every == 0:
                    print("Saving to {}...".format(ckpt))
                    saver.save(session, ckpt, global_step=global_step)

                # 验证
                if global_step % config.eval_every == 0:
                    # 训练集 F1 EM
                    train_f1, train_em, train_loss = self.get_batch_f1_em(session, batch)
                    self.add_summary(train_summary_writer, train_f1, 'train/F1', global_step)
                    self.add_summary(train_summary_writer, train_em, 'train/EM', global_step)
                    print("train: f1:{},em:{}".format(train_f1, train_em))

                    # 验证集 loss F1 EM
                    dev_loss, dev_f1, dev_em = self.validate(session)
                    self.add_summary(valid_summary_writer, dev_loss, "dev/loss", global_step)
                    self.add_summary(valid_summary_writer, dev_f1, 'dev/F1', global_step)
                    self.add_summary(valid_summary_writer, dev_em, 'dev/EM', global_step)
                    print("dev: loss:{}, f1:{},em:{}".format(dev_loss, dev_f1, dev_em))

                    # 更新最好的模型
                    if best_F1 is None or best_F1 < dev_f1:
                        print("New Better Model！ Saving...")
                        best_model_saver.save(session, best_model_ckpt, global_step=global_step)
                        best_F1 = dev_f1

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
