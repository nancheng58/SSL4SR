# coding=utf-8

"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import os
import tensorflow as tf


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=64,
                 num_hidden_layers=2,
                 num_attention_heads=2,
                 intermediate_size=256,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.2,
                 attention_probs_dropout_prob=0.2,
                 max_position_embeddings=50,
                 preference_size=3,
                 type_vocab_size=16,
                 initializer_range=0.02):
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.preference_size=preference_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    def __init__(self,config,gpu,is_training,
                 input_ids_x, input_ids_y,
                 label_ids_x, label_ids_y,
                 input_mask_x, input_mask_y,
                 scope=None):

        config = copy.deepcopy(config)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.cov_loss_wt = 1.0  # used for the ratio of coverage-loss

        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        with tf.variable_scope(scope, default_name="bert"):

            self.embedding_table = tf.get_variable(name="word_embeddings",
                                                   shape=[config.vocab_size, config.hidden_size],
                                                   initializer=create_initializer(config.initializer_range))

            self.train_both(input_ids_x=input_ids_x, input_ids_y=input_ids_y,
                             label_ids_x=label_ids_x, label_ids_y=label_ids_y,
                             input_mask_x=input_mask_x, input_mask_y=input_mask_y,
                             config=config, is_training=is_training)


    def train_each(self, input_ids, label_ids, input_mask, embedding_table, config, is_training, Sx, labelx, maskx):

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if not is_training:
            input_ids = Sx
            label_ids = labelx
            input_mask = maskx

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            # Perform embedding lookup on the word ids.
            embedding_output = embedding_lookup(input_ids=input_ids,
                                                embedding_table=embedding_table,
                                                embedding_size=config.hidden_size)

            # Add positional embeddings and token type embeddings, then layer normalize and perform dropout.
            embedding_output = embedding_postprocessor(input_tensor=embedding_output,
                                                        use_position_embeddings=True,
                                                        position_embedding_name="position_embeddings",
                                                        initializer_range=config.initializer_range,
                                                        max_position_embeddings=config.max_position_embeddings,
                                                        preference_size=config.preference_size,
                                                        dropout_prob=config.hidden_dropout_prob)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # This converts a 2D mask of [b, L] to a 3D mask of [b, L, L] which is used for the attention scores.
            padding_mask = create_padding_mask(input_ids, input_mask)
            print('padding mask:', padding_mask)

            encoder_mask = get_mask_from_encoder(embedding_output, config.preference_size)
            print('encoder mask:', encoder_mask)

            # Run the stacked transformer.
            all_encoder_layers = transformer_model(
                input_tensor=embedding_output,
                attention_mask=padding_mask,
                encoder_mask=encoder_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

        sequence_output = all_encoder_layers[-1]  #[b, L, h]
        print('encoder output:', sequence_output)

        with tf.variable_scope("identification", reuse=tf.AUTO_REUSE):
            identify_mask = get_mask_for_identi(sequence_output, preference_size=config.preference_size)
            print('identification mask:', identify_mask)  #[B,L,L]

            preference, atten_prob = identify_model(encoder_output=sequence_output,
                                                  mask=identify_mask,
                                                  preference_size=config.preference_size,
                                                  hidden_size=config.hidden_size,
                                                  num_attention_heads=config.num_attention_heads,
                                                  padding_mask=padding_mask,
                                                  item_size=config.max_position_embeddings)
            print('identification:', preference)  # [B,P,h]
            preference_all = tf.reduce_sum(preference, axis=1)  # [B,h]

            print('multiple att distributions on items:', atten_prob) # [[B,I],[B,I],[B,I]...]

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

            log_probs = decoder(preference_all=preference_all,
                                 embedding_table=embedding_table,
                                 config=config) #[B,V]

            pred_loss = predict_loss(log_probs=log_probs,
                                      label_ids=label_ids,
                                      embedding_table=embedding_table)
            print('pred_loss:', pred_loss)

            c_loss = coverage_loss(atten_prob=atten_prob,
                                   input_mask=input_mask,
                                   config=config, batch_size=batch_size)
            print('coverage_loss:', c_loss)

            loss_all = tf.add(pred_loss, self.cov_loss_wt * c_loss, name='loss_all')
            print('model-loss-all:', loss_all)


        return log_probs, loss_all, preference, pred_loss, c_loss

    def train_both(self, input_ids_x, input_ids_y, label_ids_x, label_ids_y, input_mask_x, input_mask_y, config, is_training):

        self.log_probs_x, self.loss_x, self.preference_x,\
        self.pred_loss_x, self.c_loss_x = self.train_each(input_ids=input_ids_x,
                                               input_mask=input_mask_x,
                                               label_ids=label_ids_x,
                                               embedding_table=self.embedding_table,
                                               config=config,
                                               is_training=is_training,
                                               Sx=input_ids_x,
                                               labelx=label_ids_x,
                                               maskx=input_mask_x)
        # preference_x:[b,P,h]
        print('preference-x:', self.preference_x)
        print('pred-loss-x:', self.pred_loss_x)
        print('cover-loss-x:', self.c_loss_x)

        self.log_probs_y, self.loss_y, self.preference_y,\
        self.pred_loss_y, self.c_loss_y = self.train_each(input_ids=input_ids_y,
                                               input_mask=input_mask_y,
                                               label_ids=label_ids_y,
                                               embedding_table=self.embedding_table,
                                               config=config,
                                               is_training=is_training,
                                               Sx=input_ids_x,
                                               labelx=label_ids_x,
                                               maskx=input_mask_x)
        # preference_y:[b,P,h]
        print('preference-y:', self.preference_y)
        print('pred-loss-y:', self.pred_loss_y)
        print('cover-loss-y:', self.c_loss_y)

        with tf.variable_scope("separation"):
            C_x, C_y, U_x, U_y = separation(self.preference_x, self.preference_y, config.preference_size)

        with tf.variable_scope("permutation"):
            self.new_preference_x, self.new_preference_y = permutation(C_x, C_y, U_x, U_y, config)
            print('new-preference-x:', self.new_preference_x) #[b,P,h]
            print('new-preference-y:', self.new_preference_y) #[b,P,h]

        with tf.variable_scope("common_loss"):

            loss1 = tf.losses.mean_squared_error(C_x, C_y)
            loss2 = tf.losses.mean_squared_error(self.new_preference_x, self.preference_x)
            loss3 = tf.losses.mean_squared_error(self.new_preference_y, self.preference_y)
            self.common_loss = loss1 + loss2 + loss3
            print('common-loss:', self.common_loss)

        with tf.variable_scope("new_pred_loss", reuse=tf.AUTO_REUSE):
            new_x_all = tf.reduce_sum(self.new_preference_x, axis=1)
            log_probs_x = decoder(preference_all=new_x_all,
                                  embedding_table=self.embedding_table,
                                  config=config)
            pred_loss_x = predict_loss(log_probs=log_probs_x,
                                       label_ids=label_ids_x,
                                       embedding_table=self.embedding_table)
            print('new-prediction-loss-x:', pred_loss_x)

            new_y_all = tf.reduce_sum(self.new_preference_y, axis=1)
            log_probs_y = decoder(preference_all=new_y_all,
                                  embedding_table=self.embedding_table,
                                  config=config)
            pred_loss_y = predict_loss(log_probs=log_probs_y,
                                       label_ids=label_ids_y,
                                       embedding_table=self.embedding_table)
            print('new-prediction-loss-y:', pred_loss_y)

            self.new_pred_loss = pred_loss_x + pred_loss_y
            print('new-prediction-loss:', self.new_pred_loss)

        self.loss_ssl = 10*self.common_loss + 0.1*self.new_pred_loss
        print('total SSL-loss:', self.loss_ssl)

        self.loss_learning = 0.1*self.loss_x + 0.1*self.loss_y + self.loss_ssl
        print('total learning loss:', self.loss_learning)



def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
                inputs=input_tensor,
                begin_norm_axis=-1,
                begin_params_axis=-1,
                scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""

    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     embedding_table,
                     embedding_size=128,):
    # This function assumes that the input is of shape [batch_size, seq_length, num_inputs].
    # If the input is a 2D tensor of shape [batch_size, seq_length], we reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])

    return output


def embedding_postprocessor(input_tensor,
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            preference_size=3,
                            dropout_prob=0.1):

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_position_embeddings:
        full_position_embeddings = tf.get_variable(
            name=position_embedding_name,
            shape=[seq_length, width],
            initializer=create_initializer(initializer_range))
        position_embeddings = tf.reshape(full_position_embeddings,[1,seq_length, width])
        output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)

    return output


def create_padding_mask(from_tensor, to_mask):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask

def get_mask_from_encoder(input_tensor,preference_size):

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    item_size = seq_length - preference_size

    p_mask_mat = tf.ones([preference_size, seq_length])
    print('preference-mask-matrix:', p_mask_mat)  # [p,L]

    p_zero = tf.zeros([item_size, preference_size])
    i_one = tf.ones([item_size, item_size])
    i_mask_mat = tf.concat([p_zero,i_one],axis=1)
    print('item-mask-matrix:', i_mask_mat) #[i,L]

    mask_mat = tf.concat([p_mask_mat, i_mask_mat], axis=0)
    print('mask-matrix:', mask_mat) #[L,L]

    mask_mat = tf.expand_dims(mask_mat,axis=0) #[1,L,L]
    mask_mat = tf.tile(mask_mat, [batch_size,1,1]) #[B,L,L]
    print('expand-dim-mask-mat:', mask_mat) #[B,L,L]

    return mask_mat


def attention_layer(from_tensor,
                    to_tensor,
                    encoder_mask,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError("When passing in rank 2 tensors to attention_layer, the values for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
                    from_tensor_2d,
                    num_attention_heads * size_per_head,
                    activation=query_act,
                    name="query",
                    kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
                    to_tensor_2d,
                    num_attention_heads * size_per_head,
                    activation=key_act,
                    name="key",
                    kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
                    to_tensor_2d,
                    num_attention_heads * size_per_head,
                    activation=value_act,
                    name="value",
                    kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length, size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size,
                                     num_attention_heads, to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None: # padding mask
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions,
        # this operation will create a tensor which is 0.0 for positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is effectively the same as removing these entirely.
        attention_scores += adder

    # encoder mask
    encoder_mask = tf.expand_dims(encoder_mask, axis=[1])
    adder = (1.0 - tf.cast(encoder_mask, tf.float32)) * -10000.0
    attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer, attention_probs


def transformer_model(input_tensor,
                      encoder_mask,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and forth from a 3D tensor to a 2D tensor.
    # Re-shapes are normally free on the GPU/CPU but may not be free on the TPU,
    # so we want to minimize them to help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
            layer_input = prev_output # [B*L, hidden]

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head,attention_probs = attention_layer(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            encoder_mask=encoder_mask,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=
                            attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                                        attention_output,
                                        hidden_size,
                                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output,hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                                        attention_output,
                                        intermediate_size,
                                        activation=intermediate_act_fn,
                                        kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                                        intermediate_output,
                                        hidden_size,
                                        kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

def get_mask_for_identi(input_tensor,preference_size):

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    item_size = seq_length - preference_size

    eys = tf.eye(preference_size)
    ones = tf.ones([preference_size,item_size])
    p_mask = tf.concat([eys,ones],axis=1) #[p,L]

    i2p = tf.zeros([item_size, preference_size])
    i2i = tf.ones([item_size, item_size])
    i_mask = tf.concat([i2p, i2i], axis=1) #[I,L]

    mask_mat = tf.concat([p_mask, i_mask],axis=0) #[L,L]

    mask_mat = tf.expand_dims(mask_mat, axis=0)  # [1,L,L]
    mask_mat = tf.tile(mask_mat, [batch_size, 1, 1])  # [B,L,L]
    print('expand-dim-mask-mat:', mask_mat)  # [B,L,L]

    return mask_mat


def identify_model(encoder_output, mask, preference_size, hidden_size,num_attention_heads,padding_mask,item_size):
    # encoder_output:[B,L,h], mask:[B,L,L]
    input_shape = get_shape_list(encoder_output, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    attention_head_size = int(hidden_size / num_attention_heads)

    layer_input = encoder_output
    print('before identify:', layer_input)
    preference_L, att_prob = attention_layer(from_tensor=layer_input, to_tensor=layer_input,
                                   encoder_mask=mask, attention_mask=padding_mask,
                                   size_per_head=attention_head_size,
                                   num_attention_heads=num_attention_heads,
                                    do_return_2d_tensor=False, batch_size=batch_size,
                                    from_seq_length=seq_length, to_seq_length=seq_length)
    print('identification on length:',preference_L) #[B,L,h]
    print('attention prob:', att_prob)  # [B,N,L,L]
    att_prob = tf.reduce_mean(att_prob, axis=1)  # [B,L,L]
    att_prob = tf.slice(att_prob, [0, 0, preference_size], [batch_size, preference_size, item_size])
    print('multiple att-distributions are:', att_prob)  # [B,P,I]

    att_dis_list = [tf.squeeze(s) for s in
                    tf.split(att_prob, num_or_size_splits=preference_size, axis=1)]

    preference = tf.slice(preference_L,[0,0,0],[batch_size, preference_size, width])
    print('multiple preferences are:', preference) #[B,P,h]

    return preference, att_dis_list

def decoder(preference_all, embedding_table, config):

    input_tensor = tf.layers.dense(preference_all,
                                   units=config.hidden_size,
                                   activation=gelu,
                                   kernel_initializer=create_initializer(config.initializer_range))
    input_tensor = layer_norm(input_tensor)  #[b,h]

    # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
    output_bias = tf.get_variable("output_bias",
                                  shape=[embedding_table.shape[0]],
                                  initializer=tf.zeros_initializer())
    print('output_bais:', output_bias)
    logits = tf.matmul(input_tensor, embedding_table, transpose_b=True)
    print(logits)  #[B,V]
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits, -1)

    return log_probs

def predict_loss(log_probs, label_ids, embedding_table):

    with tf.variable_scope("prediction_loss"):
        label_ids = tf.reshape(label_ids, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=embedding_table.shape[0], dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        pred_loss = tf.reduce_mean(per_example_loss, name='pred_loss')

    return pred_loss

def coverage_loss(atten_prob, input_mask, config, batch_size):

    with tf.variable_scope("coverage_loss"):
        coverage_mask = tf.cast(tf.slice(input_mask, [0, config.preference_size],
                                         [batch_size, config.max_position_embeddings]), tf.float32)  # [b,I]
        coverage = tf.zeros_like(atten_prob[0])  # shape (b, attn_length). Initial coverage is zero.
        print('coverage-vector:', coverage)
        covlosses = []
        # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
        for a in atten_prob:
            covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
            covlosses.append(covloss)
            coverage += a  # update the coverage vector
        coverage_loss = _mask_and_avg(covlosses, coverage_mask)
    return coverage_loss

def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member

    return tf.reduce_mean(values_per_ex)  # overall average

def separation(Px, Py, preference_size):
    # P1:[B,P,h], P2:[B,P,h]

    interact_matrix = dot_matrix(Px, Py, preference_size) #[b,K,K]
    print('interact_matrix:', interact_matrix)
    # row-softmax
    atten_row = tf.nn.softmax(interact_matrix)  # [b,K,K]
    Cy = tf.matmul(atten_row, Py)
    print(Cy)  #[b,K,h]
    one = tf.ones_like(atten_row)
    Uy = tf.matmul((one-atten_row), Py)
    print(Uy)  #[b,K,h]

    # col-softmax
    atten_col = tf.nn.softmax(interact_matrix, axis=1)
    Cx = tf.matmul(atten_col, Px, transpose_a=True)
    print(Cx) #[b,K,h]
    Ux = tf.matmul((one-atten_col), Px, transpose_a=True)
    print(Ux) #[b,K,h]

    return Cx, Cy, Ux, Uy

def permutation(C_x, C_y, U_x, U_y, config):
    # input are all [b,K,h]

    new_x = tf.concat([C_y, U_x, C_y*U_x], axis=-1) #[b,K,3h]
    new_x = tf.layers.dense(new_x, units=config.hidden_size)
    print('new-x:', new_x) #[b,K,h]

    new_y = tf.concat([C_x, U_y, C_x*U_y], axis=-1) #[b,K,3h]
    new_y = tf.layers.dense(new_y, units=config.hidden_size)
    print('new-y:', new_y) #[b,K,h]

    return new_x, new_y


def dot_matrix(P1_output, P2_output,preference_size):
    # P1_output=[b,K,h], P2_output=[b,K,h]

    P1_output_ext = tf.expand_dims(P1_output, axis=2)  # encoder_output_ext=[b,K,1,h]
    P1_output_ext = tf.tile(P1_output_ext, [1, 1, tf.shape(P2_output)[1], 1])  # encoder_output_ext=[b,K,K,h]

    P2_output_ext = tf.expand_dims(P2_output, axis=1)  # transfer_output_ext=[b,1,K,h]
    P2_output_ext = tf.tile(P2_output_ext, [1, tf.shape(P1_output)[1], 1, 1])  # transfer_output_ext=[b,K,K,h]

    dot = tf.concat([P1_output_ext, P2_output_ext, P1_output_ext * P2_output_ext],
                    axis=-1)  # dot=[b,K,K,h*3]
    print('dot:', dot)
    dot = tf.layers.dense(dot, 1,
                          activation=None,
                          use_bias=False,
                          kernel_initializer=create_initializer(0.02))  # dot=[b,K,K,1]
    print('dot-dense:', dot) #[b,K,K,1]
    # dot = tf.squeeze(dot)
    dot = tf.reshape(dot,[-1, preference_size, preference_size])
    print('dot-squeeze:', dot) #[b,K,K]

    return dot

# *********************** function **********************************

def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`.
        If this is specified and the `tensor` has a different rank, and exception will be thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
