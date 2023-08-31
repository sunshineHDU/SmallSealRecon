from pickle import TRUE
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple

from tensorflow.python.ops.gen_math_ops import arg_max
from tensorflow.python.ops.math_ops import argmax
from ops import *
from dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from utils import *
from trans import *
import tensorflow.contrib as tc

# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss", "cheat_loss", "t_loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
# EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "embedding"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class UNet(object):
    def __init__(self, experiment_dir=None, experiment_id=60, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                 Lcategory_penalty=1.0):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        # input pic size
        self.input_width = input_width
        # output pic size
        self.output_width = output_width
        # the size to input into the generator
        self.generator_dim = generator_dim
        # ? the size to input into the discriminator
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.input_filters = 3
        self.output_filters = 3

        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def encoder(self, images, is_training, reuse=False, name="encoder"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, e6, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False, name="decoder"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="g_d%d_deconv" % layer)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            # decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            # d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=None, do_concat=False)
            # d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=None, do_concat=False)
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator2B(self, images, is_training, reuse=False):
        e8, e6, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse, name="encoder")
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse, name="decoderA2B")
        return output, e8, e6

    def generator2A(self, images, is_training, reuse=False):
        e8, e6, enc_layers = self.encoder(images, is_training=is_training, reuse=True, name="encoder")
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse, name="decoderB2A")
        return output, e8, e6

    def discriminator(self, image, is_training, reuse=False, name="discriminator"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # input img -> [batch, 256, 256, 6]
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")

            return tf.nn.sigmoid(fc1)

    def embedding(self, inp, vocab_size, zero_pad=True):
        """When the `zero_pad` flag is on, the first row in the embedding lookup table is
        fixed to be an all-zero vector, corresponding to the '<pad>' symbol."""
        embed_size = self.d_model
        embed_lookup = tf.get_variable("embed_lookup", [vocab_size, embed_size], tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())

        if zero_pad:
            assert self._pad_id == 0
            embed_lookup = tf.concat((tf.zeros(shape=[1, self.d_model]), embed_lookup[1:, :]), 0)

        out = tf.nn.embedding_lookup(embed_lookup, inp)
        return out

    def _positional_encoding_embedding(self, inp):
        batch_size, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_embedding'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
            return self.embedding(pos_ind, seq_len, zero_pad=False)  # [batch, seq_len, d_model]

    def _positional_encoding_sinusoid(self, inp):
        """
        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        batch, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_sinusoid'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch, 1])

            # Compute the arguments for sin and cos: pos / 10000^{2i/d_model})
            # Each dimension is sin/cos wave, as a function of the position.
            pos_enc = np.array([
                [pos / np.power(10000., 2. * (i // 2) / self.d_model) for i in range(self.d_model)]
                for pos in range(seq_len)
            ])  # [seq_len, d_model]

            # Apply the cosine to even columns and sin to odds.
            pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
            pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, d_model]
            if True:
                lookup_table = tf.concat((tf.zeros(shape=[1, self.d_model]), lookup_table[1:, :]),
                                         0)

            out = tf.nn.embedding_lookup(lookup_table, pos_ind)  # [batch, seq_len, d_model]
            return out

    def positional_encoding(self, inp):
        pos_enc = self._positional_encoding_sinusoid(inp)
        return pos_enc

    def preprocess(self, inp, inp_vocab, scope):
        # Pre-processing: embedding + positional encoding
        # Output shape: [batch, seq_len, d_model]
        with tf.variable_scope(scope):
            out = self.embedding(inp, inp_vocab, zero_pad=True) + self.positional_encoding(inp)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
        return out

    def layer_norm(self, inp):
        return tc.layers.layer_norm(inp, center=True, scale=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, d_model)
            K (tf.tensor): of shape (h * batch, k_size, d_model)
            V (tf.tensor): of shape (h * batch, k_size, d_model)
            q: 4*4
            mask (tf.tensor): of shape (h * batch, q_size, k_size)
        """

        d = self.d_model // self.h
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        out = tf.layers.dropout(out, training=self._is_training)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out

    def multihead_attention(self, query, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, d_model)
            q_size:4*4
            memory (tf.tensor): of shape (batch, m_size, d_model)
            mask (tf.tensor): shape (batch, q_size, k_size)
        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            Q = tf.layers.dense(query, self.d_model, activation=tf.nn.relu)
            K = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)
            V = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)

            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, d_model/num_heads]
            Q_split = tf.concat(tf.split(Q, self.h, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.h, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.h, axis=2), axis=0)
            mask_split = tf.tile(mask, [self.h, 1, 1])

            # Apply scaled dot product attention
            out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_split)

            # Merge the multi-head back to the original shape
            out = tf.concat(tf.split(out, self.h, axis=0), axis=2)  # [bs, q_size, d_model]

            # The final linear layer and dropout.
            # out = tf.layers.dense(out, self.d_model)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

        return out

    def feed_forwad(self, inp, scope='ff'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).
        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            # out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
            # out = tf.layers.dense(out, self.d_model, activation=None)

            # by default, use_bias=True
            out = tf.layers.conv1d(out, filters=self.d_ff, kernel_size=1, activation=tf.nn.relu)
            out = tf.layers.conv1d(out, filters=self.d_model, kernel_size=1)

        return out

    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]
        # tf.cast -> to change the data type in tf.not_equal(inp, self._pad_id)
        mask = tf.cast(tf.not_equal(inp, self._pad_id), tf.float32)  # mask '<pad>'
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
        return mask

    def construct_padding_mask_input(self):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        # todo
        one = tf.ones([self.batch_size, self.seq_len])
        # zero = tf.ones([self.batch_size,self.seq_len-self.seq_len])
        # mask = tf.cast(tf.concat([one, zero], 1), tf.float32)
        # mask = tf.tile(tf.expand_dims(one, 1), [1, 1, self.seq_len])
        # tf.tile -> tanping tf.expand_dims(one, 1) with shape [1, self.seq_len, 1]
        mask = tf.tile(tf.expand_dims(one, 1), [1, self.seq_len, 1])
        return mask

    def construct_autoregressive_mask(self, target):
        """
        Args: Original target of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len].
        """
        batch_size, seq_len = target.shape.as_list()

        tri_matrix = np.zeros((seq_len, seq_len))
        tri_matrix[np.tril_indices(seq_len)] = 1

        mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
        masks = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1])  # copies
        return masks

    def encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forword
            out = self.layer_norm(out + self.multihead_attention(out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def t_encoder(self, inp, input_mask, scope='encoder'):
        """
        Args:
            inp (tf.tensor): shape (batch, seq_len, embed_size)
            input_mask (tf.tensor): shape (batch, seq_len, seq_len)
            scope (str): name of the variable scope.
        """
        out = inp  # now, (batch, seq_len, embed_size)
        with tf.variable_scope(scope):
            for i in range(self.num_enc_layers):
                out = self.encoder_layer(out, input_mask, f'enc_{i}')
        return out

    def decoder_layer(self, target, enc_out, input_mask, target_mask, scope):
        out = target
        with tf.variable_scope(scope):
            out = self.layer_norm(out + self.multihead_attention(
                out, mask=target_mask, scope='self_attn'))
            out = self.layer_norm(out + self.multihead_attention(
                out, memory=enc_out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def t_decoder(self, target, enc_out, input_mask, target_mask, scope='decoder'):
        out = target
        with tf.variable_scope(scope):
            # todo num_enc_layers = 6,暂时先6吧
            num_enc_layers = 6
            for i in range(num_enc_layers):
                out = self.decoder_layer(out, enc_out, input_mask, target_mask, f'dec_{i}')
        return out

    def label_smoothing(self, inp):
        """
        From the paper: "... employed label smoothing of epsilon = 0.1. This hurts perplexity,
        as the model learns to be more unsure, but improves accuracy and BLEU score."
        Args:
            inp (tf.tensor): one-hot encoding vectors, [batch, seq_len, vocab_size]
        """
        vocab_size = inp.shape.as_list()[-1]
        smoothed = (1.0 - self.ls_epsilon) * inp + (self.ls_epsilon / vocab_size)
        return smoothed

    def transformer(self, encoded, is_training, reuse=False, name="transformer"):
        # 4*4 512
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            positionEmbedded = positionEmbedding(lastBatch=tf.shape(encoded)[0], scope="positionEmbedding")
            embedded = encoded + positionEmbedded
            # 维度[batch_size, sequence_length, embedding_size]
            multiHeadAtt = multiheadAttention(rawKeys=encoded, queries=embedded, keys=embedded,
                                              scope="multiheadAttention")
            # 维度[batch_size, sequence_length, embedding_size]
            # todo 这个序列我觉得可能需要改
            embeddedWords = feedForward(multiHeadAtt, [64, 512], scope="feedForward")
            outputs = tf.reshape(embeddedWords, [-1, 4 * 4 * 512])
            return fcn(outputs, scope="fcn")

    def rtn(self, encoded, is_training, reuse=False, name="rtn"):
        # 4*4 512
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # Add the offset on the input and target sentences.

            # For the input we remove the starting <s> to keep the seq len consistent.

            # For the decoder input, we remove the last element, as no more future prediction
            # is gonna be made based on it.
            dec_inp = self.raw_target[:, :-1]  # starts with <s>
            dec_target = self.raw_target[:, 1:]  # starts with the first word
            dec_target_ohe = tf.one_hot(dec_target, depth=self.target_vocab)
            if self.use_label_smoothing:
                dec_target_ohe = self.label_smoothing(dec_target_ohe)

            # The input mask only hides the <pad> symbol.
            input_mask = self.construct_padding_mask_input()

            # The target mask hides both <pad> and future words.
            target_mask = self.construct_padding_mask(dec_inp)
            target_mask *= self.construct_autoregressive_mask(dec_inp)

            # Input embedding + positional encoding
            # inp_embed = self.preprocess(enc_inp, self.input_vocab, "input_preprocess")
            # enc_out = self.encoder(inp_embed, input_mask)

            # Target embedding + positional encoding
            dec_inp_embed = self.preprocess(dec_inp, self.target_vocab, "target_preprocess")
            dec_out = self.t_decoder(dec_inp_embed, encoded, input_mask, target_mask)

            # Make the prediction out of the decoder output.
            logits = tf.layers.dense(dec_out, self.target_vocab)  # [batch, target_vocab]
            return logits

    def build_model(self, is_training=True):
        # ?
        self.seq_len = 4 * 4
        # todo: need to be changed. 408
        # the max length of the predicted radical sequence
        self.target_vocab = 409
        # ?
        self.input_vocab = 4 * 4
        # ?
        self.ls_epsilon = 0.1
        # ?
        self.use_label_smoothing = True
        # number used to padding the radical seq
        self._pad_id = 0
        # ?
        self.d_model = 512  # dim
        self.drop_rate = 0.1
        # ?
        self.h = 8  # head nums
        self.d_ff = 512  # todo
        self._is_training = is_training
        # three pic concat in channel dim
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.input_width, self.input_width,
                                         self.input_filters + self.output_filters + self.output_filters],
                                        name='real_A_and_B_images')
        # target images
        self.real_B = self.real_data[:, :, :, :self.input_filters]
        # source images
        self.real_A = self.real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]
        # ? maybe the standard font pic
        self.targetA = self.real_data[:, :, :, self.input_filters * 2:self.input_filters * 3]

        # the label of the batch (what is the label?)
        self.label = tf.placeholder(tf.int32, [self.batch_size], name='label')

        # self._raw_input = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len + 1], name='raw_input')

        # the radical seq of the input, len is 17, start with '2', end with '3', padded with '0'
        self.raw_target = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len + 1], name='raw_target')

        self.fake_A2B, encoded_real_ABe8, encoded_real_ABe6 = self.generator2B(self.real_A, is_training=is_training,
                                                                               reuse=False)
        self.fake_B2B, encoded_real_BBe8, encoded_real_BBe6 = self.generator2B(self.real_B, is_training=is_training,
                                                                               reuse=True)
        self.fake_B2A, encoded_real_BAe8, encoded_real_BAe6 = self.generator2A(self.real_B, is_training=is_training,
                                                                               reuse=False)
        self.fake_A2A, encoded_real_AAe8, encoded_real_AAe6 = self.generator2A(self.real_A, is_training=is_training,
                                                                               reuse=True)

        # _, _, encoded_fake_ABe6 = self.generator2B(self.fake_A2B, is_training=is_training, reuse=True)
        # _, _, encoded_fake_BAe6 = self.generator2B(self.fake_B2A, is_training=is_training, reuse=True)

        _, encoded_target_Ae8, encoded_target_Ae6 = self.generator2B(self.targetA, is_training=is_training, reuse=True)

        self.dropoutKeepProb = 0.5

        # RTN part  -------------------> start
        # For the decoder input, we remove the last element, as no more future prediction
        # is gonna be made based on it.
        # ? related to raw_target, but I don't know what is raw_target
        # raw_target: size == [self.batch_size, self.seq_len + 1]
        dec_inp = self.raw_target[:, :-1]  # starts with <s>
        dec_target = self.raw_target[:, 1:]  # starts with the first word
        # create one_hot data, dec_target as indicate, change every pos in indicate to len(depth) one hot tensor,
        # and [indicate[val]] is 1
        dec_target_ohe = tf.one_hot(dec_target, depth=self.target_vocab)
        if self.use_label_smoothing:
            dec_target_ohe = self.label_smoothing(dec_target_ohe)

        # Make the prediction out of the decoder output.
        logits = self.rtn(tf.reshape(encoded_real_ABe6, [self.batch_size, 4 * 4, 512]), is_training=is_training,
                          reuse=False, name="rtn")
        # tf.argmax -> find max in axis.
        self._output = tf.argmax(logits, axis=-1, output_type=tf.int32)

        target_not_pad = tf.cast(tf.not_equal(dec_target, self._pad_id), tf.float32)
        self.predictions = tf.reduce_sum(
            tf.cast(tf.equal(self._output, dec_target), tf.float32) * target_not_pad /
            tf.cast(tf.reduce_sum(target_not_pad), tf.float32)
        )

        # transformer_loss
        t_loss = (tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=dec_target_ohe)))

        # optim = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
        #                                beta1=0.9, beta2=0.98, epsilon=1e-9)
        # self._train_op = optim.minimize(self._loss)

        # RTN part  -------------------> end

        real_AB = tf.concat([self.targetA, self.real_B], 3)
        fake_AB = tf.concat([self.targetA, self.fake_A2B], 3)
        fake_BB = tf.concat([self.targetA, self.fake_B2B], 3)
        real_BA = tf.concat([self.real_B, self.real_A], 3)
        fake_BA = tf.concat([self.real_B, self.fake_B2A], 3)
        fake_AA = tf.concat([self.real_B, self.fake_A2A], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting reuse to True
        # real_AB, real_AB_logits = self.discriminator(real_AB, is_training=is_training, reuse=False,
        #                                              name="discriminator2B")
        # fake_AB, fake_AB_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True,
        #                                              name="discriminator2B")
        # fake_BB, fake_BB_logits = self.discriminator(fake_BB, is_training=is_training, reuse=True,
        #                                              name="discriminator2B")
        # real_BA, real_BA_logits = self.discriminator(real_BA, is_training=is_training, reuse=False,
        #                                              name="discriminator2A")
        # fake_BA, fake_BA_logits = self.discriminator(fake_BA, is_training=is_training, reuse=True,
        #                                              name="discriminator2A")
        # fake_AA, fake_AA_logits = self.discriminator(fake_AA, is_training=is_training, reuse=True,
        #                                              name="discriminator2A")
        real_AB_logits = self.discriminator(real_AB, is_training=is_training, reuse=False,
                                                     name="discriminator2B")
        fake_AB_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True,
                                                     name="discriminator2B")
        fake_BB_logits = self.discriminator(fake_BB, is_training=is_training, reuse=True,
                                                     name="discriminator2B")
        real_BA_logits = self.discriminator(real_BA, is_training=is_training, reuse=False,
                                                     name="discriminator2A")
        fake_BA_logits = self.discriminator(fake_BA, is_training=is_training, reuse=True,
                                                     name="discriminator2A")
        fake_AA_logits = self.discriminator(fake_AA, is_training=is_training, reuse=True,
                                                     name="discriminator2A")

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        # encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        # count MSE(square + reduce_mean)
        const_loss = (tf.reduce_mean(tf.square(encoded_real_ABe8 - encoded_real_BAe8))
                      + tf.reduce_mean(tf.square(encoded_real_ABe8 - encoded_target_Ae8))
                      + tf.reduce_mean(tf.square(encoded_real_ABe6 - encoded_target_Ae6))
                      + tf.reduce_mean(tf.square(encoded_real_BAe8 - encoded_target_Ae8))
                      + tf.reduce_mean(tf.square(encoded_real_BAe6 - encoded_target_Ae6))
                      + tf.reduce_mean(tf.square(encoded_real_ABe6 - encoded_real_BAe6))) * self.Lconst_penalty

        # binary real/fake loss
        # tf.nn.sigmoid_cross_entropy_with_logits -> compute the cross_entropy of logit with sigmoid
        d_ab_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_AB_logits,
                                                                                labels=tf.ones_like(real_AB_logits)))
        d_ab_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_AB_logits,
                                                                                labels=tf.zeros_like(fake_AB_logits)))
        d_bb_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_BB_logits,
                                                                                labels=tf.zeros_like(fake_BB_logits)))
        d_ba_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_BA_logits,
                                                                                labels=tf.ones_like(real_BA_logits)))
        d_ba_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_BA_logits,
                                                                                labels=tf.zeros_like(fake_BA_logits)))
        d_aa_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_AA_logits,
                                                                                labels=tf.zeros_like(fake_AA_logits)))
        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * (tf.reduce_mean(tf.abs(self.fake_A2B - self.real_B)) + tf.reduce_mean(
            tf.abs(self.fake_B2B - self.real_B)) + tf.reduce_mean(
            tf.abs(self.fake_A2A - self.targetA)) + tf.reduce_mean(tf.abs(self.fake_B2A - self.targetA)))
        # total variation loss
        width = self.output_width
        # tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
        #            + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty

        # maximize the chance generator fool the discriminator
        cheat_ab_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_AB_logits,
                                                                               labels=tf.ones_like(fake_AB_logits)))
        cheat_ba_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_BA_logits,
                                                                               labels=tf.ones_like(fake_BA_logits)))
        cheat_aa_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_AA_logits,
                                                                               labels=tf.ones_like(fake_AA_logits)))
        cheat_bb_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_BB_logits,
                                                                               labels=tf.ones_like(fake_BB_logits)))
        cheat_loss = cheat_ab_loss + cheat_ba_loss + cheat_aa_loss + cheat_bb_loss

        d_loss = d_ab_loss_real + d_ab_loss_fake + d_ba_loss_real + d_ba_loss_fake + d_aa_loss_fake + d_bb_loss_fake
        g_loss = cheat_loss + l1_loss + const_loss

        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_ab_loss_real + d_ba_loss_real)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake",
                                                d_ab_loss_fake + d_ba_loss_fake + d_aa_loss_fake + d_bb_loss_fake)
        cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        const_loss_summary = tf.summary.scalar("const_loss", const_loss)
        d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)

        d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
                                             d_loss_summary])
        g_merged_summary = tf.summary.merge([cheat_loss_summary, l1_loss_summary,
                                             const_loss_summary,
                                             g_loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=self.real_data)

        loss_handle = LossHandle(d_loss=d_loss,
                                 g_loss=g_loss,
                                 const_loss=const_loss,
                                 cheat_loss=cheat_loss,
                                 t_loss=t_loss,
                                 l1_loss=l1_loss)

        # eval_handle = EvalHandle(encoder=encoded_real_A,
        #                          generator=fake_B,
        #                          target=real_B,
        #                          source=real_A)

        summary_handle = SummaryHandle(d_merged=d_merged_summary,
                                       g_merged=g_merged_summary)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        # setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e" in var.name)]

        return g_vars, d_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'encoder' in var.name or "decoder" in var.name or "g_" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        # eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images, labels, epoch, idx):
        # print(true)
        y_labels = []
        for label in labels:
            nums = label.split(',')
            y_label = []
            for num in nums:
                y_label.append(int(num))
            y_labels.append(y_label)
        fake_BA, fake_AB, fake_BB, fake_AA, pre = self.sess.run(
            [self.fake_B2A, self.fake_A2B, self.fake_B2B, self.fake_A2A, self.predictions],
            feed_dict={self.real_data: input_images, self.raw_target: y_labels}
        )
        # accu = accuracy(pre, true)
        print("The val data's accuracy: %.5f" % pre)
        # merged_pair = np.concatenate([fake_AA, fake_BA], axis=1)
        # merged_pair = np.concatenate([merged_pair, fake_BB], axis=1)
        # merged_pair = np.concatenate([merged_pair, fake_AB], axis=1)
        # save_images(merged_pair, [self.batch_size, 1],
        #             './{}/AA_BA_BB_AB{:02d}_{:04d}.jpg'.format('dir', epoch, idx))
        save_path = os.path.join(self.experiment_dir, 'sample')
        save_images(fake_AB, [self.batch_size, 1], os.path.join(save_path, 'AB_{:02d}_{:04d}.jpg'.format(epoch, idx)))
        save_images(fake_BA, [self.batch_size, 1], os.path.join(save_path, 'BA_{:02d}_{:04d}.jpg'.format(epoch, idx)))
        save_images(fake_BB, [self.batch_size, 1], os.path.join(save_path, 'BB_{:02d}_{:04d}.jpg'.format(epoch, idx)))
        save_images(fake_AA, [self.batch_size, 1], os.path.join(save_path, 'AA_{:02d}_{:04d}.jpg'.format(epoch, idx)))
        # input_handle, loss_handle, summary_handle = self.retrieve_handles()
        # fake_images, real_images, \
        # d_loss, g_loss, l1_loss = self.sess.run([eval_handle.generator,
        #                                          loss_handle.d_loss,
        #                                          loss_handle.g_loss,
        #                                          loss_handle.l1_loss],
        #                                         feed_dict={
        #                                             input_handle.real_data: input_images
        #                                         })
        # return fake_images, real_images, d_loss, g_loss, l1_loss

    def val_accuracy(self, val_iter):
        count = 0
        acc = 0
        for labels, source_imgs in val_iter:
            y_labels = []
            for label in labels:
                print(label)
                nums = label.split(',')
                y_label = []
                for num in nums:
                    y_label.append(int(num))
                y_labels.append(y_label)
            fake_BA, fake_AB, fake_BB, fake_AA, pre, outputs = self.sess.run(
                [self.fake_B2A, self.fake_A2B, self.fake_B2B, self.fake_A2A, self.predictions, self._output],
                feed_dict={self.real_data: source_imgs, self.raw_target: y_labels})
            # count += accuracy_num(pre, labels)
            count += pre * len(labels)
            for output in outputs:
                #     # data = [argmax(out) for out in output]
                print(output)

        print("The all val data's accuracy: %d" % count)

    def validate_model(self, val_iter, epoch, step):
        # val_batch_iter = data_provider.get_val_iter(self.batch_size)
        # ei -> epoch i
        # counter -> batch i + 1
        labels, images = next(val_iter)
        self.generate_fake_samples(images, labels, epoch, step)
        # fake_imgs, real_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(images, labels)
        # print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss: %.5f" % (d_loss, g_loss, l1_loss))

        # merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        # merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        # merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        # model_id, _ = self.get_model_id_and_dir()

        # model_sample_dir = os.path.join(self.sample_dir, model_id)
        # if not os.path.exists(model_sample_dir):
        #     os.makedirs(model_sample_dir)

        # sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        # misc.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, model_dir, save_dir):
        data_provider = TrainDataProvider(source_obj)
        train_batch_iter = data_provider.get_test_iter(self.batch_size, False)
        # source_provider = InjectDataProvider(source_obj)

        # if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
        #     embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
        #     source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        # else:
        #     source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def check_dir(path_dir):
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)

        path_save_fake_pic = os.path.join(save_dir, 'fake')
        check_dir(path_save_fake_pic)
        for each in ['AA', 'AB', 'BA', 'BB']:
            check_dir(os.path.join(path_save_fake_pic, each))

        path_gt_save = os.path.join(save_dir, 'gt')
        check_dir(path_gt_save)

        # def save_imgs(imgs, count):
        #     p = os.path.join(save_dir, "%04d_1.jpg" % count)
        #     save_concat_images(imgs, img_path=p)
        #     print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        num = 0

        with open(os.path.join(save_dir, 'fake.txt'), 'w', encoding='UTF-8') as fake_txt_file:
            with open(os.path.join(save_dir, 'gt.txt'), 'w', encoding='UTF-8') as gt_txt_file:
                for labels, source_imgs in train_batch_iter:
                    y_labels = []
                    for label in labels:
                        nums = label.split(',')
                        gt_txt_file.write(label + '\n')
                        y_label = []
                        for num in nums:
                            y_label.append(int(num))
                        y_labels.append(y_label)
                    fake_BA, fake_AB, fake_BB, fake_AA, pre, outputs = self.sess.run(
                                [self.fake_B2A, self.fake_A2B, self.fake_B2B, self.fake_A2A, self.predictions, self._output],
                                feed_dict={self.real_data: source_imgs, self.raw_target: y_labels})
                    # accu = accuracy(pre, true)
                    print("The val data's accuracy: %.5f" % pre)
                    save_images(fake_AB, [self.batch_size, 1], os.path.join(path_save_fake_pic, 'AB', f'AB_{count:04d}.jpg'))
                    save_images(fake_BA, [self.batch_size, 1], os.path.join(path_save_fake_pic, 'BA', f'AB_{count:04d}.jpg'))
                    save_images(fake_BB, [self.batch_size, 1], os.path.join(path_save_fake_pic, 'BB', f'AB_{count:04d}.jpg'))
                    save_images(fake_AA, [self.batch_size, 1], os.path.join(path_save_fake_pic, 'AA', f'AB_{count:04d}.jpg'))

                    save_images(source_imgs[:, :, :, :3], [self.batch_size, 1],
                                os.path.join(path_gt_save, f'real_ch_{count:04d}.jpg'))
                    save_images(source_imgs[:, :, :, 3:6], [self.batch_size, 1],
                                os.path.join(path_gt_save, f'real_seal_{count:04d}.jpg'))

                    count += 1
                    for output in outputs:
                        gt_txt_file.write(', '.join([str(each) for each in output]))
                        gt_txt_file.write('\n')

            #     y_labels = []
                #     for label in labels:
                #         nums = label.split(',')
                #         y_label = []
                #         for each_num in nums:
                #             y_label.append(int(each_num))
                #         y_labels.append(y_label)
                #     fake_BA, fake_AB, fake_BB, fake_AA, pre = self.sess.run(
                #         [self.fake_B2A, self.fake_A2B, self.fake_B2B, self.fake_A2A, self.predictions],
                #         feed_dict={self.real_data: source_imgs, self.raw_target: y_labels})
                #     # acc = accuracy(pre, labels)
                #     # count += accuracy_num(pre, labels)
                #     count += pre * len(labels)
                #     print("%.5f" % pre)
                #
                #     save_images(fake_AB, [self.batch_size, 1],
                #                 f'./{save_dir}/{num:05d}.jpg')
                #
                #     # save_images(self.real_B, [self.batch_size, 1], f'./{save_dir}/{num:05d}.jpg')
                #     num += 1
                #     # save_images(fake_AB, [self.batch_size, 1],
                #     #         './{}/AB_{:02d}_{:04d}.jpg'.format('val', labels[0], labels[0]))
                #     # save_images(fake_BA, [self.batch_size, 1],
                #     #         './{}/BA_{:02d}_{:04d}.jpg'.format('val', labels[0], labels[0]))
                #     # save_images(realA, [self.batch_size, 1],
                #     #         './{}/A_{:02d}_{:04d}.jpg'.format('val', labels[0], labels[0]))
                #     # save_images(realB, [self.batch_size, 1],
                #     #         './{}/B_{:02d}_{:04d}.jpg'.format('val', labels[0], labels[0]))
                # print(count)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append(x * (1. - alpha) + alpha * y)

            interpolated = np.asarray(interpolated, dtype=np.float32)
            return interpolated

        def filter_embedding_vars(var):
            var_name = var.name
            if var_name.find("embedding") != -1:
                return True
            if var_name.find("inst_norm/shift") != -1 or var_name.find("inst_norm/scale") != -1:
                return True
            return False

        embedding_vars = filter(filter_embedding_vars, tf.trainable_variables())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the embedding at the end
        embedding_snapshot = list()
        for e_var in embedding_vars:
            val = e_var.eval(session=self.sess)
            embedding_snapshot.append((e_var, val))
            t = _interpolate_tensor(val)
            op = tf.assign(e_var, t, validate_shape=False)
            print("overwrite %s tensor" % e_var.name, "old_shape ->", e_var.get_shape(), "new shape ->", t.shape)
            self.sess.run(op)

        source_provider = InjectDataProvider(source_obj)
        input_handle, _, eval_handle, _ = self.retrieve_handles()
        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                generated, = self.sess.run([eval_handle.generator],
                                           feed_dict={
                                               input_handle.real_data: source_imgs,
                                               input_handle.embedding_ids: labels
                                           })
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))
        # restore the embedding variables
        print("restore embedding values")
        for var, val in embedding_snapshot:
            op = tf.assign(var, val, validate_shape=False)
            self.sess.run(op)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500):
        # get all variables
        g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)
        t_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-4).minimize(
            loss_handle.t_loss)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val_iter(self.batch_size)

        # save the model
        saver = tf.train.Saver(max_to_keep=3)
        # draw a graph
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                labels, batch_images = batch
                y_labels = []
                for label in labels:
                    nums = label.split(',')
                    y_label = []
                    for num in nums:
                        y_label.append(int(num))
                    y_labels.append(y_label)
                # #类别
                # num_classes = 3692
                # #one_hot编码
                # one_hot_codes = np.eye(num_classes)

                # one_hot_labels = []
                # for label in labels:
                # #将连续的整型值映射为one_hot编码
                #     one_hot_label = one_hot_codes[label]
                #     one_hot_labels.append(one_hot_label)

                # one_hot_labels = np.array(one_hot_labels)

                # Optimize D
                _, _, batch_d_loss, transformer_loss, d_summary = self.sess.run([d_optimizer, t_optimizer,
                                                                                 loss_handle.d_loss,
                                                                                 loss_handle.t_loss,
                                                                                 summary_handle.d_merged],
                                                                                feed_dict={
                                                                                    real_data: batch_images,
                                                                                    self.raw_target: y_labels,
                                                                                    learning_rate: current_lr
                                                                                })
                # Optimize G
                _, _, batch_g_loss = self.sess.run([g_optimizer, t_optimizer, loss_handle.g_loss],
                                                   feed_dict={
                                                       real_data: batch_images,
                                                       self.raw_target: y_labels,
                                                       learning_rate: current_lr
                                                   })
                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, _, batch_g_loss, cheat_loss, transformer_loss, \
                const_loss, l1_loss, pre, g_summary = self.sess.run([g_optimizer, t_optimizer,
                                                                     loss_handle.g_loss,
                                                                     loss_handle.cheat_loss,
                                                                     loss_handle.t_loss,
                                                                     loss_handle.const_loss,
                                                                     loss_handle.l1_loss,
                                                                     self.predictions,
                                                                     summary_handle.g_merged],
                                                                    feed_dict={
                                                                        real_data: batch_images,
                                                                        self.raw_target: y_labels,
                                                                        learning_rate: current_lr
                                                                    })
                passed = time.time() - start_time
                # acc = accuracy(pre, labels)
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, transformer_loss: %.5f, g_loss: %.5f, " + \
                             "cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f, accuracy: %.5f"
                print(log_format % (ei, bid, total_batches, passed, batch_d_loss, transformer_loss, batch_g_loss,
                                    cheat_loss, const_loss, l1_loss, pre))
                summary_writer.add_summary(d_summary, counter)
                summary_writer.add_summary(g_summary, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    self.validate_model(val_batch_iter, ei, counter)

                # if counter % (2) == 0:
                #     # sample the current model states with val data
                #     test_batch_iter = data_provider.get_test_iter(self.batch_size)
                #     self.val_accuracy(test_batch_iter)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)
