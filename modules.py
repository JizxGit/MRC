# _*_ coding:utf8 _*_
import tensorflow as tf


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

        v_P = []
        for i in range(context_len):

            ## As in the paper
            u_Q = question_encoding  # [batch_size, q_length, 2*hidden_size_encoder]
            u_P = context_encoding  # [batch_size, context_length, 2*hidden_size_encoder]

            W_uQ_uQ = self.matrix_multiplication(u_Q, self.W_uQ)  # [batch_size, q_len, hidden_size_qp]
            print("Shape W_uQ_cal", W_uQ_uQ.shape)

            cur_batch_size = tf.shape(context_encoding)[0]
            concat_u_Pi = tf.concat([tf.reshape(u_P[:, i, :], [cur_batch_size, 1, 2 * self.hidden_size_encoder])] * question_len, 1)

            # u_Pi_slice = tf.reshape(u_P[:, i, :], [cur_batch_size, 1, 2 * self.hidden_size_encoder])

            print("Shape Concat", concat_u_Pi)
            W_uP_uPi = self.matrix_multiplication(concat_u_Pi, self.W_uP)  # [batch_size, 1, hidden_size_qp]

            print("Shape W_uP_cal", W_uP_uPi.shape)  # [batch_size, 1, hidden_size_qp]

            if i == 0:
                tanh_qp = tf.tanh(W_uQ_uQ + W_uP_uPi)  # [batch_size, q_length, hidden_size_qp]
            else:

                concat_v_Pi = tf.concat([tf.reshape(v_P[i - 1], [cur_batch_size, 1, self.hidden_size_qp])] * question_len, 1)
                # v_Pi_slice = tf.reshape(v_P[i - 1], [cur_batch_size, 1, self.hidden_size_qp])

                W_vP_vPi = self.matrix_multiplication(concat_v_Pi, self.W_vP)
                print("Shape W_vP_cal", W_vP_vPi.shape)  # [batch_size, 1, hidden_size_qp]
                tanh_qp = tf.tanh(W_uQ_uQ + W_uP_uPi + W_vP_vPi)

            print("Shape tanh", tanh_qp.shape)
            # Calculate si = vT*tanh
            s_i_qp = self.matrix_multiplication(tanh_qp, tf.reshape(self.v_t, [-1, 1]))  # [batch_size, q_length, 1]
            print("Shape s_i", s_i_qp.shape)

            s_i_qp = tf.squeeze(s_i_qp, axis=2)  # [batch_size, q_length]. Same shape as values Mask

            # print("Shape values mask", values_mask.shape)

            _, a_i_qp = masked_softmax(s_i_qp, values_mask, 1)  # [batch_size, q_length]
            print("Shape a_i_qp", a_i_qp.shape)

            a_i_qp = tf.expand_dims(a_i_qp, axis=1)  # [batch_size, 1,  q_length]
            c_i_qp = tf.reduce_sum(tf.matmul(a_i_qp, u_Q), 1)  # [batch_size, 2 * hidden_size_encoder]

            print("Shape c_i", c_i_qp.shape)

            # gate

            slice = u_P[:, i, :]

            print("Shape slice", slice)
            u_iP_c_i = tf.concat([slice, c_i_qp], 1)
            print("Shape u_iP_c_i", u_iP_c_i.shape)  # [batch_size, 4*hidden_size_encoder]

            g_i = tf.sigmoid(tf.matmul(u_iP_c_i, self.W_g_QP))
            print("Shape g_i", g_i.shape)  # batch_size, 4*hidden_size_encoder]

            u_iP_c_i_star = tf.multiply(u_iP_c_i, g_i)  # batch_size, 4*hidden_size_encoder]

            print("Shape u_iP_c_i_star", u_iP_c_i_star.shape)

            self.QP_state = self.QP_cell.zero_state(batch_size=cur_batch_size, dtype=tf.float32)

            # QP_attention
            with tf.variable_scope("QP_attention"):
                if i > 0: tf.get_variable_scope().reuse_variables()
                output, self.QP_state = self.QP_cell(u_iP_c_i_star, self.QP_state)
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


class Bidaf_output_layer(object):
    def __init__(self, context_len, concat_len):
        self.context_len = context_len
        self.concat_len = concat_len

        pass

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
