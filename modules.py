# _*_ coding:utf8 _*_
import tensorflow as tf
import numpy as np

class RNNEncoder(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        # 双向rnn
        self.rnn_cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell_fw)
        self.rnn_cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell_bw)

    def build_graph(self, inputs, mask, scope_name):
        with tf.variable_scope(scope_name):
            inputs_len = tf.reduce_sum(mask, axis=1)
            (fw_output, bw_output), last_state = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, inputs_len,
                                                                                 dtype=tf.float32)
            output = tf.concat([fw_output, bw_output], 2)
            output = tf.nn.dropout(output, self.keep_prob)
            return output


def create_rnn_graph(rnn_layer_num, hidden_size, x, x_mask, scope_name):
    outs = []
    inputs_len = tf.reduce_sum(x_mask, axis=1)
    for i in range(rnn_layer_num):
        f_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        f_cell = tf.nn.rnn_cell.DropoutWrapper(f_cell)
        b_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        b_cell = tf.nn.rnn_cell.DropoutWrapper(b_cell)
        outputs, final_output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, x,
                                                                       dtype=tf.float32,
                                                                       sequence_length=inputs_len,
                                                                       scope=scope_name + '_rnn{}'.format(i))

        # outputs: A tuple (output_fw, output_bw)
        x = tf.concat(outputs, axis=-1)  # 将前后向的输出拼接起来
        outs.append(x)

    res = tf.concat(outs, axis=-1)  # 最后将每一层的输出拼接在一起
    return res


class BasicAttention(object):
    '''
    最基本的attention，
    '''

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        # self.key_vec_size = key_vec_size
        # self.value_vec_size = value_vec_size

    def build_graph(self, keys, values, values_mask):
        '''
        key对value进行attention，得到[M,N]的attention矩阵，然后按行进行softmask，最后再乘以value，
        :param keys: shape=[batch_size,M,H]
        :param values:shape=[batch_size,N,H]
        :param values_mask: [batch_size,N]
        :return: [batch_size,M,H]
        '''
        values_t = tf.transpose(values, perm=[0, 2, 1])
        attn_matrix = tf.matmul(keys, values_t)  # [bacth_size,M,N]
        attn_matrix_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, N)
        _, attn_dist = masked_softmax(attn_matrix, attn_matrix_mask, 2)
        output = tf.matmul(attn_dist, values)  # [batch_size,M,N]=[bacth_size,M,N] * [batch_size,N,H]
        output = tf.nn.dropout(output, self.keep_prob)
        return attn_dist, output


class Bidaf(object):
    def __init__(self, keep_prob, vec_size):
        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.w = tf.get_variable('w', [vec_size * 3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    def build_graph(self, context, question, c_mask, q_mask):
        with tf.variable_scope('BiDAF'):
            context_expand = tf.expand_dims(context, axis=2)  # [batch_size,context_len,1,2h]
            question_expand = tf.expand_dims(question, axis=1)  # [batch_size,1,ques_len,2h]
            # 按位乘法，tf的*支持广播操作
            c_elem_wise_q = context_expand * question_expand  # [batch_size,context_len,ques_len,2h]

            # 复制向量 都组成[batch_size,context_len,ques_len,2h]的shape
            context_tile = tf.tile(context_expand, [1, 1, question.get_shape().as_list()[1], 1])
            question_tile = tf.tile(question_expand, [1, context.get_shape().as_list()[1], 1, 1])
            print("shape:{}".format(context.get_shape().as_list()[1]))
            print("shape context_tile:{}".format(context_tile.shape))
            print("shape question_tile:{}".format(question_tile.shape))
            print("shape c_elem_wise_q:{}".format(c_elem_wise_q.shape))
            concated_input = tf.concat([context_tile, question_tile, c_elem_wise_q], -1)
            print("concat_shape:{}".format(concated_input.shape))

            similarity_matrix = tf.reduce_sum(concated_input * self.w, axis=3)

            # Context - to - query Attention
            similarity_mask = tf.expand_dims(q_mask, axis=1)
            _, c2q_dist = masked_softmax(similarity_matrix, similarity_mask, 2)  # [batch,context_len,ques_len]
            c2q_attn = tf.matmul(c2q_dist, question)  # [batch,context_len,2h]
            print("c2q_attn shape: {}".format(c2q_attn.shape))

            # Query - to - context Attention.
            T_max = tf.reduce_max(similarity_matrix, axis=2)  # [batch,context_len]
            print("T_max shape: {}".format(T_max.shape))
            _, q2c_dist = masked_softmax(T_max, c_mask, 1)  # [batch,context_len]
            print("q2c_dist shape: {}".format(q2c_dist.shape))

            # 为了进行矩阵乘法，进行扩展一维[1,m]*[m,2h]=[1,2h]
            # q2c_dist_expand =  [batch,1,context_len]
            q2c_attn = tf.matmul(tf.expand_dims(q2c_dist, axis=1), context)  # [batch,1,2h]
            # context_len = context.get_shape().as_list()[1]

            # context * c2q_attn=[batch,context_len,2h]
            # context * q2c_attn=[batch,context_len,2h] 按位乘 [batch,context_len,2h]
            output = tf.concat([c2q_attn, context * c2q_attn, context * q2c_attn], axis=2)
            output = tf.nn.dropout(output, self.keep_prob)
            return output


class Attention_Match_RNN(object):
    """Module for Gated Attention and Self Matching from paper - https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
      Apply gated attention recurrent network for both query-passage matching and self matching networks
        Based on the explanation in http://web.stanford.edu/class/cs224n/default_project/default_project_v2.pdf
    """

    def create_weights(self, size_in, size_out, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in, size_out),
                               initializer=tf.contrib.layers.xavier_initializer())

    def create_vector(self, size_in, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in),
                               initializer=tf.contrib.layers.xavier_initializer())

    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p

    def __init__(self, keep_prob, hidden_size_encoder, hidden_size_qp, hidden_size_sm):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          inp_vec_size: size of the input vector
        """
        self.keep_prob = keep_prob
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_qp = hidden_size_qp
        self.hidden_size_sm = hidden_size_sm

        # For QP attention
        self.W_uQ = self.create_weights(2 * self.hidden_size_encoder, self.hidden_size_qp, name='W_uQ')
        self.W_uP = self.create_weights(2 * self.hidden_size_encoder, self.hidden_size_qp, name='W_uP')
        self.W_vP = self.create_weights(self.hidden_size_qp, self.hidden_size_qp, name='W_vP')
        self.W_g_QP = self.create_weights(4 * self.hidden_size_encoder, 4 * self.hidden_size_encoder, name='W_g_QP')
        self.v_t = self.create_vector(self.hidden_size_qp, name='v_t')

        # For self attention
        self.W_vP_self = self.create_weights(self.hidden_size_qp, self.hidden_size_sm, name='W_vP_self')
        self.W_vP_hat_self = self.create_weights(self.hidden_size_qp, self.hidden_size_sm, name='W_vP_hat_self')
        self.W_g_self = self.create_weights(2 * self.hidden_size_qp, 2 * self.hidden_size_qp, name='W_g_self')
        self.v_t_self = self.create_vector(self.hidden_size_sm, name='v_t_self')

        self.QP_cell = tf.contrib.rnn.GRUCell(self.hidden_size_qp)  # initiate GRU cell
        self.QP_cell = tf.contrib.rnn.DropoutWrapper(self.QP_cell,
                                                     input_keep_prob=self.keep_prob)  # added dropout wrapper

        self.SM_fw = tf.contrib.rnn.GRUCell(self.hidden_size_sm)  # initiate GRU cell
        self.SM_fw = tf.contrib.rnn.DropoutWrapper(self.SM_fw,
                                                   input_keep_prob=self.keep_prob)  # added dropout wrapper

        self.SM_bw = tf.contrib.rnn.GRUCell(self.hidden_size_sm)  # initiate GRU cell
        self.SM_bw = tf.contrib.rnn.DropoutWrapper(self.SM_bw,
                                                   input_keep_prob=self.keep_prob)  # added dropout wrapper

    def build_graph_qp_matching(self, context_encoding, question_encoding, values_mask, context_mask, context_len, question_len):
        """
        Implement question passage matching from R-Net
        """
        u_Q = question_encoding
        u_P = context_encoding

        v_P = []  # gated rnn的结果
        # 不能使用动态的rnn了，因为每个t的输入，都要用到t-1时刻的输出
        cur_batch_size = tf.shape(context_encoding)[0]
        self.QP_state = self.QP_cell.zero_state(batch_size=cur_batch_size, dtype=tf.float32)

        for i in range(context_len):
            # tanh里的第1部分
            W_uQ_uQ = self.matrix_multiplication(u_Q, self.W_uQ)

            # tanh里的第2部分
            u_iP = u_P[:, i, :]
            W_uP_iP = self.matrix_multiplication(u_iP, self.W_uP)

            # tanh里的第3部分
            if i == 0:
                tanh_qp = tf.tanh(W_uQ_uQ + W_uP_iP)
            else:
                v_t_1_P = v_P[i - 1]
                W_vP_vPi = self.matrix_multiplication(v_t_1_P, self.W_vP)
                tanh_qp = tf.tanh(W_uQ_uQ + W_uP_iP + W_vP_vPi)

            # tanh 外的vt
            s_i = tf.squeeze(self.matrix_multiplication(tanh_qp, self.v_t), axis=2)  # [batch_size,q,1]->[batch_size,q]
            _, a_i = masked_softmax(s_i, values_mask, 1)  # [batch_size,q]
            a_i_qp = tf.expand_dims(a_i, axis=1)  # [batch_size,1,q]
            c_i = tf.reduce_sum(tf.matmul(a_i_qp, u_Q), axis=1)  # [batch,2*hidden_size_encoder]

            # gate
            concat_ip_c_i = tf.concat([u_iP, c_i], axis=1)
            g_t = tf.sigmoid(tf.matmul(self.W_g_QP, concat_ip_c_i))
            concat_ip_c_i_star = tf.multiply(g_t, concat_ip_c_i)

            # 进行rnn输出
            with tf.variable_scope("QP_attention"):
                if i > 0: tf.get_variable_scope().reuse_variables()
                output, self.QP_state = self.QP_cell(concat_ip_c_i_star, self.QP_state)
                v_P.append(output)

        v_P = tf.stack(v_P, 1)
        v_P = tf.nn.dropout(v_P, self.keep_prob)
        print("Shape v_P", v_P.shape)  # [batch_size, context_len, hidden_size_qp]
        return v_P

    def build_graph_sm_matching(self, context_encoding, question_encoding, values_mask, context_mask, context_len,
                                question_len, v_P):
        """
        Implement self matching from R-Net
        """

        ## Start Self Matching
        sm = []
        u_Q = question_encoding  # [batch_size, q_length, 2*hidden_size_encoder]
        u_P = context_encoding  # [batch_size, context_length, 2*hidden_size_encoder]

        for i in range(context_len):
            W_vP_vPself = self.matrix_multiplication(v_P, self.W_vP_self)  # [batch_size, context_len, hidden_size_sm]

            print("Shape W_vP_vPself", W_vP_vPself.shape)

            cur_batch_size = tf.shape(v_P)[0]

            # slice_v_iP = tf.reshape(v_P[:, i, :], [cur_batch_size, 1, self.hidden_size_qp])

            concat_v_iP = tf.concat(
                [tf.reshape(v_P[:, i, :], [cur_batch_size, 1, self.hidden_size_qp])] * context_len, 1)
            W_vP_vPhat_self = self.matrix_multiplication(concat_v_iP,
                                                         self.W_vP_hat_self)  # [batch_size, 1, hidden_size_sm]

            print("Shape W_vP_vPhat_self", W_vP_vPhat_self.shape)

            tanh_sm = tf.tanh(W_vP_vPself + W_vP_vPhat_self)  # [batch_size, context_len, hidden_size_sm]

            print("Shape tanh", tanh_sm.shape)

            # Calculate si = vT*tanh
            s_i_sm = self.matrix_multiplication(tanh_sm,
                                                tf.reshape(self.v_t_self, [-1, 1]))  # [batch_size, context_len, 1]
            print("Shape S_i", s_i_sm.shape)

            s_i_sm = tf.squeeze(s_i_sm, axis=2)  # [batch_size, context_len]

            _, a_i_sm = masked_softmax(s_i_sm, context_mask, 1)  # [batch_size, context_len]

            print("Shape a_i_sm", a_i_sm.shape)  # [batch_size, context_len]

            a_i_sm = tf.expand_dims(a_i_sm, axis=1)
            c_i_sm = tf.reduce_sum(tf.matmul(a_i_sm, v_P), 1)  # [batch_size, hidden_size_qp]

            print("Shape c_i", c_i_sm.shape)

            # gate
            slice_vP = v_P[:, i, :]
            v_iP_c_i = tf.concat([slice_vP, c_i_sm], 1)  # [batch_size, 2*hidden_size_qp]
            print("Shape v_iP_c_i", v_iP_c_i.shape)

            g_i_self = tf.sigmoid(tf.matmul(v_iP_c_i, self.W_g_self))  # [batch_size, 2*hidden_size_qp]
            print("Shape g_i_self", g_i_self.shape)

            v_iP_c_i_star = tf.multiply(v_iP_c_i, g_i_self)

            print("Shape v_iP_c_i_star", v_iP_c_i_star.shape)  # [batch_size, 2*hidden_size_qp]

            sm.append(v_iP_c_i_star)
        sm = tf.stack(sm, 1)
        unstacked_sm = tf.unstack(sm, context_len, 1)

        self.SM_fw_state = self.SM_fw.zero_state(batch_size=cur_batch_size, dtype=tf.float32)
        self.SM_bw_state = self.SM_bw.zero_state(batch_size=cur_batch_size, dtype=tf.float32)

        with tf.variable_scope('Self_match') as scope:
            SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(self.SM_fw, self.SM_bw,
                                                                                           unstacked_sm,
                                                                                           dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1)
        h_P = tf.nn.dropout(h_P, self.keep_prob)

        print("Shape h_P", h_P.shape)  # [batch_size, context_len, 2*hidden_size_sm]

        return h_P


class Answer_Pointer(object):
    def create_weights(self, size_in, size_out, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in, size_out),
                               initializer=tf.contrib.layers.xavier_initializer())

    def create_vector(self, size_in, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in),
                               initializer=tf.contrib.layers.xavier_initializer())

    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p

    def __init__(self, keep_prob, hidden_size_encoder, question_len, hidden_size_attn):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size_encoder = hidden_size_encoder
        self.keep_prob = keep_prob
        self.hidden_size_attn = hidden_size_attn
        self.question_len = question_len

        ## Initializations for question pooling
        self.W_ruQ = self.create_weights(2 * self.hidden_size_encoder, self.hidden_size_encoder, name='W_ruQ')
        self.W_vQ = self.create_weights(self.hidden_size_encoder, self.hidden_size_encoder, name='W_vQ')

        ## Same size as question hidden
        self.W_VrQ = self.create_weights(self.question_len, self.hidden_size_encoder, name='W_VrQ')
        self.v_qpool = self.create_vector(self.hidden_size_encoder, name='v_qpool')

        ## Initializations for answer pointer
        self.W_hP = self.create_weights(self.hidden_size_attn, 2 * self.hidden_size_encoder, name='W_hP')
        self.W_ha = self.create_weights(2 * self.hidden_size_encoder, 2 * self.hidden_size_encoder, name='W_ha')

        self.v_ptr = self.create_vector(2 * self.hidden_size_encoder, name='v_ptr')

        self.ans_ptr_cell = tf.contrib.rnn.GRUCell(2 * self.hidden_size_encoder)  # initiate GRU cell
        self.ans_ptr_cell = tf.contrib.rnn.DropoutWrapper(self.ans_ptr_cell, input_keep_prob=self.keep_prob)  # added dropout wrapper

    def question_pooling(self, question_encoding, values_mask):
        ## Question Pooling as suggested in R-Net Paper

        u_Q = question_encoding

        # tanh的第一部分
        W_ruQ_u_Q = self.matrix_multiplication(u_Q, self.W_ruQ)  # [batch_size, q_length, hidden_size_encoder]
        # print("Shape W_ruQ_u_Q", W_ruQ_u_Q.shape)
        # tanh的第二部分
        W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_vQ)  # [ q_length, hidden_size_encoder]
        cur_batch_size = tf.shape(u_Q)[0]
        W_vQ_V_rQ = tf.expand_dims(W_vQ_V_rQ, axis=0)
        # 相加做tanh
        tanh_qpool = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)  # [batch_size, q_length, hidden_size_encoder]
        s_i_qpool = self.matrix_multiplication(tanh_qpool, tf.reshape(self.v_qpool, [-1, 1]))  # [batch_size, q_len, 1]

        # 第二个公式，做softmax
        s_i_qpool = tf.squeeze(s_i_qpool, axis=2)  # [batch_size, q_length]. Same shape as values Mask
        _, a_i_qpool = masked_softmax(s_i_qpool, values_mask, 1)  # [batch_size, q_length]
        # 第三个公式，做pooling
        a_i_qpool = tf.expand_dims(a_i_qpool, axis=1)  # [batch_size, 1,  q_length]
        r_Q = tf.reduce_sum(tf.matmul(a_i_qpool, u_Q), 1)  # [batch_size, 2 * hidden_size_encoder]

        r_Q = tf.nn.dropout(r_Q, self.keep_prob)
        print(' shape of r_Q', r_Q.shape)  # [batch_size, 2 * hidden_size_encoder]
        return r_Q

    def build_graph_answer_pointer(self, context_hidden, ques_encoding, values_mask, context_mask, context_len):

        h_P = context_hidden
        r_Q = self.question_pooling(ques_encoding, values_mask)
        h_a = None  # pointer network 的输出 last hidden state
        p = []  # 记录开始位置，结束位置的，经过softmax后的
        logits = []
        cur_batch_size = tf.shape(ques_encoding)[0]
        for i in range(2):
            # 第一个公式
            # tanh 第一部分
            W_hp_h_p = self.matrix_multiplication(h_P, self.W_hP)

            # tanh 第二部分
            # 公式9中的h_t-1_a，初始化时用r_Q，然后才是用经过 pointer network 得到的last hidden state
            if i == 0:
                h_t_1_a = r_Q
            else:
                h_t_1_a = h_a

            concat_h_i1a = tf.concat([tf.reshape(h_t_1_a, [cur_batch_size, 1, 2 * self.hidden_size_encoder])] * context_len, 1)
            W_ha_h_i1a = self.matrix_multiplication(concat_h_i1a, self.W_ha)

            tanh = tf.tanh(W_hp_h_p + W_ha_h_i1a)
            s_t = self.matrix_multiplication(tanh, tf.reshape(self.v_ptr, [-1, 1]))  # [batch_size,context_len,1]
            s_t = tf.squeeze(s_t, axis=2)

            # 第二个公式
            logits_ptr, a_t = masked_softmax(s_t, context_mask, 1)  # [batch_size,context_len]

            # 第三个公式，不进行argmax，这是外部函数的事情
            p.append(a_t)
            logits.append(logits_ptr)

            # 得到a_t后，可以进行pointer network了，也就是公式10的计算
            a_t = tf.expand_dims(a_t, 1)  # [batch_size,1,context_len]
            c_t = tf.reduce_sum(tf.matmul(a_t, h_P), 1)  # [batch_size,hidden_size]
            if i == 0:
                self.ans_ptr_state = self.ans_ptr_cell.zero_state(batch_size=cur_batch_size, dtype=tf.float32)  # TODO 论文中是说使用r_Q进行初始化？？
                h_a, _ = self.ans_ptr_cell(c_t, self.ans_ptr_state)

        return p, logits


class Bidaf_output_layer(object):
    def __init__(self, context_len, concat_len):
        self.context_len = context_len
        self.concat_len = concat_len

    def build_graph(self, blended_represent, bidaf_output, context_mask):
        w1 = tf.get_variable("w1", shape=[self.concat_len], initializer=tf.contrib.layers.xavier_initializer())  # [10h,1]
        G = tf.concat([blended_represent, bidaf_output], axis=2)  # [batch_size,context_len,10h]
        result = tf.reduce_sum(G * w1, axis=2)  # [batch_size * context_len]
        print("Shape result", result.shape)
        logits_start, prob_dist_start = masked_softmax(result, context_mask, 1)
        return logits_start, prob_dist_start


class SimpleSoftmaxLayer(object):
    def __init__(self):
        pass

    def build_graph(self, inputs, mask):
        logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1)  # shape=[batch_size,context_len,1]
        logits = tf.squeeze(logits, axis=[2])  # 将最后一维度的1去掉，shape=[batch_size,context_len]
        masked_logits, prob_dist = masked_softmax(logits, mask, 1)  # mask后的
        return masked_logits, prob_dist


def masked_softmax(logits, mask, dim):
    '''
    使用mask数组，将pad的位置变得非常小，然后与logits相加，使得pad的位置不可能被预测为最终的结果，最后才进行softmax
    :param logits:  [batch_size,seq_len]
    :param mask:
    :param dim:
    :return:
    '''
    mask_ = (1 - tf.cast(mask, 'float')) * (-1e30)  # pad的地方变得非常小【0，0，0，-1e30,-1e30】
    masked_logits = tf.add(logits, mask_)  # 然后与logits相加
    prob_distribution = tf.nn.softmax(masked_logits, dim)  # dim=1,表示对第二个维度进行softmax
    return masked_logits, prob_distribution


def SeqAttnMatch(x, y):
    """Given sequences x and y, match sequence y to each element in x.

    Args:
        x: tensor of shape batch x len1 x h
        y: tensor of shape batch x len2 x h
    Return:
        matched_seq = batch * len1 * h
    """
    len1, h = x.get_shape().as_list()[1:]
    len2 = y.get_shape().as_list()[1]

    x_proj = tf.layers.dense(tf.reshape(x, [-1, h]), h, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='proj_dense', reuse=False)
    y_proj = tf.layers.dense(tf.reshape(y, [-1, h]), h, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='proj_dense', reuse=True)

    x_proj = tf.reshape(x_proj, [-1, len1, h])
    y_proj = tf.reshape(y_proj, [-1, len2, h])
    scores = tf.einsum('ijk,ikq->ijq', x_proj, tf.transpose(y_proj, [0, 2, 1]))  # b x len1 x len2
    alpha_flat = tf.nn.softmax(tf.reshape(scores, [-1, len2]))
    alpha = tf.reshape(alpha_flat, [-1, len1, len2])
    matched_seq = tf.einsum('ijk,ikq->ijq', alpha, y)
    return matched_seq


def SelfAttn(x,x_mask):
    '''
    Self attention over a sequence.
    :param x: tensor of shape batch * len * hdim
    :return: tensor of shape batch * len
    '''
    len_, hdim = x.get_shape().as_list()[1:]
    x_flat = tf.reshape(x, [-1, hdim])  # [batch * len , hdim]
    # 建立一个全连接网络，作为 w
    weight = tf.layers.dense(x_flat, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())  # shape=[batch*len]
    weight = tf.reshape(weight, [-1, len_])  # shape=[batch,len]
    _, mask_weight =masked_softmax(weight,x_mask,1)
    return mask_weight


def bilinear_sequnce_attention(context, question):
    """ A bilinear attention layer over a sequence seq w.r.t context

    Args:
        context: 3D tensor of shape b x l x h1
        question: 2D tensor of shape b x l2

    Return:
        tensor of shape b x l with weight coefficients
    """

    len_, h1 = context.get_shape().as_list()[1:3]
    question = tf.layers.dense(question, h1, kernel_initializer=tf.contrib.layers.xavier_initializer())
    question = tf.reshape(question, [-1, h1, 1])  # b x h1 x 1
    z = tf.einsum('ijk,ikq->ijq', context, question)
    z = tf.reshape(z, [-1, len_])  # b x l
    return z


# TODO 完成 FFN
def FFN(x):
    '''
    Feed_Forward_Networks: FFN(x) = W2 ReLU(W1x+b1)+b2
    :param x: tensor of shape : batch * len * hdim
    :return:
    '''
    len_, h = x.get_shape().as_list()[1:]
    x_proj = tf.layers.dense(tf.reshape(x, [-1, h]), h, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='proj_dense', reuse=False)

    y_proj = tf.layers.dense(tf.reshape(x_proj, [-1, h]), h,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='proj_dense', reuse=False)
