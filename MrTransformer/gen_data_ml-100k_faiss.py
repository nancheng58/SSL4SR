# -*- coding: UTF-8 -*-
import os
import codecs
import faiss
import collections
import random
import sys
from collections import defaultdict
import tensorflow as tf
import six
from util import *
from vocab_p3 import *
from Jac_sparse import *
import pickle
import multiprocessing
import time
import util

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_labels, sid):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_labels = masked_lm_labels
        self.sid = sid

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (" ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_labels: %s\n" % (" ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "sid: %s\n" % (" ".join([printable_text(x) for x in self.sid]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files1(instances, max_seq_length, vocab,
                                    output_files, force_last, cross_ratio, sample_num):
    """Create TF example files from `TrainingInstance`s."""

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    id_to_ins_dic = dict(enumerate(instances))  # [seq_id: seq_ins]
    print('instance-dict-len:', len(id_to_ins_dic))

    seq_arr = []
    for (inst_index, instance) in id_to_ins_dic.items():
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        seq_arr.append(input_ids)
    arr = np.zeros([len(seq_arr), vocab.get_vocab_size()]).astype('float32')
    for seq_i, seq in enumerate(seq_arr):
        for item in seq:
            arr[seq_i, item] = 1.0
    print('build-matrix done...')
    index = faiss.IndexFlatL2(vocab.get_vocab_size())   # build the index
    index.add(arr)
    print('add index done...')
    D, I = index.search(arr, sample_num)
    print('calculate score matrix done...')
    for (inst_index, instance) in enumerate(instances):

        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        # print('tokens-id:', input_ids)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length + vocab.get_preference_token_size()
        input_ids += [0] * (max_seq_length + vocab.get_preference_token_size() - len(input_ids))
        input_mask += [0] * (max_seq_length + vocab.get_preference_token_size() - len(input_mask))  # padding mask
        assert len(input_ids) == max_seq_length + vocab.get_preference_token_size()
        assert len(input_mask) == max_seq_length + vocab.get_preference_token_size()
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        # print('label:', masked_lm_ids)
        # print('laebl-token:', instance.masked_lm_labels)
        if not force_last:
            neg_index = I[inst_index,:]
            c = 1
            for ind in neg_index:
                ins = id_to_ins_dic[ind]
                neg_ids = vocab.convert_tokens_to_ids(ins.tokens)
                assert len(neg_ids) <= max_seq_length + vocab.get_preference_token_size()
                neg_info = ins.info
                neg_labels = vocab.convert_tokens_to_ids(ins.masked_lm_labels)
                neg_mask = [1]*len(neg_ids)
                neg_ids += [0]*(max_seq_length + vocab.get_preference_token_size() - len(neg_ids))
                neg_mask += [0]*(max_seq_length + vocab.get_preference_token_size() - len(neg_mask))
                assert len(neg_ids) == max_seq_length + vocab.get_preference_token_size()
                assert len(neg_mask) == max_seq_length + vocab.get_preference_token_size()

                features = collections.OrderedDict()
                features["info_x"] = create_int_feature(instance.info)
                features["input_ids_x"] = create_int_feature(input_ids)
                features["input_mask_x"] = create_int_feature(input_mask)
                features["label_ids_x"] = create_int_feature(masked_lm_ids)

                features["info_y"] = create_int_feature(neg_info)
                features["input_ids_y"] = create_int_feature(neg_ids)
                features["input_mask_y"] = create_int_feature(neg_mask)
                features["label_ids_y"] = create_int_feature(neg_labels)
                c += 1

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))

                writers[writer_index].write(tf_example.SerializeToString())
                writer_index = (writer_index + 1) % len(writers)

                total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_instances(all_documents_raw,
                     max_seq_length,
                     rng,
                     vocab,
                     prop_sliding_window,
                     force_last=False):
    """Create `Instance`s from raw text."""
    all_documents = {}
    new_slid_data = []
    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            i_list = vocab.get_preference_token()
            all_documents[user] = [i_list + item_seq[-max_num_tokens:]]  # [[i1,i2,i3]],len=1

    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(prop_sliding_window * max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens

        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            # todo: add slide
            if len(item_seq) <= max_num_tokens:
                i_list = vocab.get_preference_token()
                all_documents[user] = [i_list + item_seq]  # [[i1,i2,i3]],len=1
            else:
                beg_idx = list(range(len(item_seq) - max_num_tokens, 0,
                                     -sliding_step))
                beg_idx.append(0)
                i_list = vocab.get_preference_token()
                all_documents[user] = [i_list + item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]


        pre_num = vocab.get_preference_token_size()
        sid = 0
        for user in all_documents.keys():
            for seq in all_documents[user]:
                if len(seq) == pre_num:
                    print("got empty seq in the train dataset:" + user)
                    continue
                # todo: add slide
                for i in range(pre_num+1, len(seq)):
                    line = []
                    line.append(user)
                    line.append(sid)
                    line.extend(seq[:i + 1])
                    new_slid_data.append(line)
                    sid += 1

    instances = []
    if force_last:  # for test
        for user in all_documents:
            instances.extend(create_instances_from_document_test(all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))

    else:  # for train
        for s_data in new_slid_data:
            instances.extend(mask_last(s_data, max_seq_length, vocab, ))
        print("num of instance:{}".format(len(instances)))

    rng.shuffle(instances)

    return instances


def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]  # [[P1,i1,i2...]]
    max_num_tokens = max_seq_length

    assert len(document) == 1 and len(document[0]) <= max_num_tokens + vocab.get_preference_token_size()

    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_labels=masked_lm_labels,
        sid=info)

    return [instance]


def mask_last(s_data, max_seq_length, vocab,):
    """Creates `TrainingInstance`s for a single document."""

    max_num_tokens = max_seq_length
    user = s_data[0]
    # print('user:',user)
    sid = s_data[1]
    tokens = s_data[2:]
    # print('document:',document)

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    assert len(tokens) >= 1 and len(tokens) <= max_num_tokens + vocab.get_preference_token_size()

    (tokens, masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
    print('tokens:', tokens)
    print('labels:', masked_lm_labels)
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_labels=masked_lm_labels,
        sid=sid)
    instances.append(instance)

    return instances


def create_masked_lm_predictions_force_last(tokens):

    assert len(tokens) > 1

    output_tokens = list(tokens[:-1])

    masked_lm_labels = [tokens[-1]]

    return (output_tokens, masked_lm_labels)


def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                prop_sliding_window,
                cross_ratio,
                sample_num,
                force_last=False,):
    # create train
    instances = create_instances(data,
                                 max_seq_length,
                                 rng,
                                 vocab,
                                 prop_sliding_window,
                                 force_last)
    print('begin to write')
    write_instance_to_example_files1(instances, max_seq_length, vocab, [output_filename], force_last, cross_ratio, sample_num)


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


if __name__ == "__main__":
    random_seed = 12345

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string("signature", 'default', "signature_name")

    flags.DEFINE_integer("max_seq_length", 50, "max sequence length.")

    flags.DEFINE_float("prop_sliding_window", 0.1, "sliding window step size.")

    flags.DEFINE_float("cross_ratio", 0.3, "control the lowest cross ratio between current seq and sampling seq.")

    flags.DEFINE_integer("sample_num", 20, "num of sample of each sequence.")

    flags.DEFINE_string("data_dir", 'data/ml-100k/', "data dir.")

    flags.DEFINE_string("dataset_name", 'ml-100k', "dataset name.")

    tf.logging.set_verbosity(tf.logging.DEBUG)

    max_seq_length = FLAGS.max_seq_length

    prop_sliding_window = FLAGS.prop_sliding_window

    cross_ratio = FLAGS.cross_ratio

    sample_num = FLAGS.sample_num

    output_dir = FLAGS.data_dir
    dataset_name = FLAGS.dataset_name
    version_id = FLAGS.signature
    print(version_id)

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    dataset = data_partition(output_dir + dataset_name + '.txt')

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
          format(len(user_train), len(user_valid), len(user_test), usernum, itemnum))

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }

    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)

    user_test_data_output = {k: [vocab.convert_tokens_to_ids(v)] for k, v in user_test_data.items()}

    print('begin to generate train')
    output_filename = output_dir + dataset_name + version_id + '.train2_p3_learning1_faiss.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        prop_sliding_window,
        cross_ratio,
        sample_num,
        force_last=False)
    print('train:{}'.format(output_filename))

    print('begin to generate test')
    output_filename = output_dir + dataset_name + version_id + '.test2_p3.tfrecord'
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        -1.0,
        cross_ratio,
        sample_num,
        force_last=True)
    print('test:{}'.format(output_filename))

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab2_p3'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his2_p3'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')