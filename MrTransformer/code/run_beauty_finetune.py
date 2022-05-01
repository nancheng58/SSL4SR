# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Run beauty with <token with P, label, len, pad_mask>"""

import os
import modeling_coverage_learning_loss
import optimization
import tensorflow as tf
import numpy as np
import sys
import pickle
import random

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", 'data/beauty/bert_config_beauty_64_p3.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", 'training data',
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", 'test data',
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("checkpointDir", 'checkpoint_file',
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_filename", 'vocab data', "vocab filename")

flags.DEFINE_string("user_history_filename", 'history data', "user history filename")

## Other parameters
flags.DEFINE_string("init_checkpoint", 'checkpoint of pretrain model(ckpt-XX)', "Initial checkpoint.") #

flags.DEFINE_integer("max_seq_length", 50,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded. Must match data generation.")
flags.DEFINE_integer("preference_size", 3,
                     "The number of preference size.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("batch_size", 512, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 3e-4, "The initial learning rate for Adam.")

flags.DEFINE_string("gpu", 'gpu-id', "gpu use.")

flags.DEFINE_integer("num_train_steps", 1000000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 6000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 1000, "How much checkpoint files to save.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("use_pop_random", True, "use pop random negative samples")


class EvalHooks(tf.estimator.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0

        self.mrr_1 = 0.0
        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.mrr_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.mrr_10 = 0.0
        self.ndcg_20 = 0.0
        self.hit_20 = 0.0
        self.mrr_20 = 0.0

        np.random.seed(12345)

        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        print("recall@1:{}, recall@5:{}, recall@10:{}， recall@20:{}, "
              "mrr@1:{}, mrr@5:{}, mrr@10:{}, mrr@20:{}, "
              "ndcg@1:{}, ndcg@5:{}, ndcg@10:{}, ndcg@20:{},valid_user:{}".
              format(self.hit_1 / self.valid_user, self.hit_5 / self.valid_user,
                     self.hit_10 / self.valid_user, self.hit_20 / self.valid_user,
                     self.mrr_1 / self.valid_user, self.mrr_5 / self.valid_user,
                     self.mrr_10 / self.valid_user, self.mrr_20 / self.valid_user,
                     self.ndcg_1 / self.valid_user, self.ndcg_5 / self.valid_user,
                     self.ndcg_10 / self.valid_user, self.ndcg_20 / self.valid_user, self.valid_user))

        with open(FLAGS.result_file, 'a+') as f:

            f.write('recall @1|5|10|20: ' + str(self.hit_1 / self.valid_user) + '\t' + str(self.hit_5 / self.valid_user) + '\t' + str(
                self.hit_10 / self.valid_user) + '\t' + str(self.hit_20 / self.valid_user) + '\t')
            f.write('mrr @1|5|10|20: ' + str(self.mrr_1 / self.valid_user) + '\t' + str(self.mrr_5 / self.valid_user) + '\t' + str(
                self.mrr_10 / self.valid_user) + '\t' + str(self.mrr_20 / self.valid_user) + '\t')
            f.write('ndcg @1|5|10|20: ' + str(self.ndcg_1 / self.valid_user) + '\t' + str(self.ndcg_5 / self.valid_user) + '\t' + str(
                self.ndcg_10 / self.valid_user) + '\t' + str(self.ndcg_20 / self.valid_user) + '\n')

    def before_run(self, run_context):
        # tf.logging.info('run before run')
        # print('run before_run')
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        # tf.logging.info('run after run')
        # print('run after run')
        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx]
            # print(masked_lm_log_probs_elem.shape) #V
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            if FLAGS.use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
                self.mrr_1 += 1
            if rank < 5:
                self.mrr_5 += 1 / (rank + 1)
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.mrr_10 += 1 / (rank + 1)
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1
            if rank < 20:
                self.mrr_20 += 1 / (rank + 1)
                self.ndcg_20 += 1 / np.log2(rank + 2)
                self.hit_20 += 1


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids_x = features["input_ids_x"]
        input_mask_x = features["input_mask_x"]
        label_ids_x = features["label_ids_x"]
        info_x = features["info_x"]

        input_ids_y = features["input_ids_y"]
        input_mask_y = features["input_mask_y"]
        label_ids_y = features["label_ids_y"]
        info_y = features["info_y"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        print('Istrain:', is_training)

        model = modeling_coverage_learning_loss.BertModel(
            config=bert_config,
            gpu=FLAGS.gpu,
            is_training=is_training,
            input_ids_x=input_ids_x, input_ids_y=input_ids_y,
            label_ids_x=label_ids_x, label_ids_y=label_ids_y,
            input_mask_x=input_mask_x, input_mask_y=input_mask_y,
        )

        # total_loss = model.loss_ssl # this loss is used for pre-train
        total_loss = model.loss_x + model.loss_y # this loss is used for finetune

        masked_lm_log_probs = model.log_probs_x

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling_coverage_learning_loss.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', input_ids_x)
            tf.add_to_collection('eval_sp', label_ids_x)
            tf.add_to_collection('eval_sp', info_x)

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_files,
                     max_seq_length,
                     preference_size,
                     is_training,
                     num_cpu_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info_x": tf.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids_x": tf.FixedLenFeature([max_seq_length + preference_size], tf.int64),
            "input_mask_x": tf.FixedLenFeature([max_seq_length + preference_size], tf.int64),
            "label_ids_x": tf.FixedLenFeature([1], tf.int64),
            "info_y": tf.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids_y": tf.FixedLenFeature([max_seq_length + preference_size], tf.int64),
            "input_mask_y": tf.FixedLenFeature([max_seq_length + preference_size], tf.int64),
            "label_ids_y": tf.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat(200)
            d = d.shuffle(buffer_size=10000)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        d = d.prefetch(1)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


if __name__ == "__main__":

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    FLAGS.checkpointDir = FLAGS.checkpointDir
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling_coverage_learning_loss.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.io.gfile.glob(input_pattern))
    print(train_input_files)

    test_input_files = []
    if FLAGS.test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
            test_input_files.extend(tf.io.gfile.glob(input_pattern))
    print(test_input_files)

    tf.compat.v1.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
        tf.compat.v1.logging.info("  %s" % input_file)

    tf.compat.v1.logging.info("*** test Input Files ***")
    for input_file in test_input_files:
        tf.compat.v1.logging.info("  %s" % input_file)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.checkpointDir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max)

    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)
    print('vocab_size:', item_size)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": FLAGS.batch_size})

    tensors_to_log = {
        "pred_loss_x": "bert/decoder/prediction_loss/pred_loss:0",
        "pred_loss_y": "bert/decoder_1/prediction_loss/pred_loss:0",
        "coverage_loss_x": "bert/decoder/coverage_loss/Mean:0",
        "coverage_loss_y": "bert/decoder_1/coverage_loss/Mean:0",
        "common_loss": "bert/common_loss/add_1:0",
        "new_pred_loss_x": "bert/new_pred_loss/prediction_loss/pred_loss:0",
        "new_pred_loss_y": "bert/new_pred_loss/prediction_loss_1/pred_loss:0",
    } # this is used to print loss


    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    if FLAGS.do_train:  # training stage
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            preference_size=FLAGS.preference_size,
            is_training=True)
        print('training data load done...')

        estimator.train(input_fn=train_input_fn,
                        max_steps=FLAGS.num_train_steps,
                        hooks=[logging_hook]) #

        # result = estimator.evaluate(
        #     input_fn=train_input_fn,
        #     steps=None,
        #     hooks=[EvalHooks()])

    # **************************************************************************************************

    # if FLAGS.do_eval:  # test stage
    #     tf.logging.info("***** Running evaluation *****")
    #     tf.logging.info("  Batch size = %d", FLAGS.batch_size)
    #
    #     eval_input_fn = input_fn_builder(
    #         input_files=test_input_files,
    #         max_seq_length=FLAGS.max_seq_length,
    #         preference_size=FLAGS.preference_size,
    #         is_training=False)
    #
    #     # tf.logging.info('special eval ops:', special_eval_ops)
    #     result = estimator.evaluate(
    #         input_fn=eval_input_fn,
    #         steps=None,
    #         hooks=[EvalHooks()])
    #
    #     output_eval_file = os.path.join(FLAGS.checkpointDir, "result.txt")
    #
    #     with tf.gfile.GFile(output_eval_file, "w") as writer:
    #         tf.logging.info("***** Eval results *****")
    #         tf.logging.info(bert_config.to_json_string())
    #         writer.write(bert_config.to_json_string() + '\n')
    #         for key in sorted(result.keys()):
    #             tf.logging.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))
