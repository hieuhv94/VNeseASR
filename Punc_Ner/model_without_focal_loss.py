import ujson
import codecs
import random
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from layers import multi_conv1d, Ner_AttentionCell, Punc_AttentionCell
from logger import get_logger, Progbar
from tensorflow_addons.text import crf_log_likelihood, viterbi_decode
from sklearn.metrics import recall_score, precision_score, f1_score
# from sklearn.metrics import confusion_matrix
#
# def p_r_f1(y_true, y_pred):
tf.compat.v1.disable_eager_execution()
def load_data(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = ujson.load(f)
    return dataset


def pad_sequences(sequences, pad=None, max_length=None):
    if pad is None:
        # 0: "PAD" for words and chars, "O" for label
        pad = 0
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
#        print(sequences)
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def process_batch_data(batch_words, batch_chars, batch_labels=None):
    b_words, b_words_len = pad_sequences(batch_words)
    b_chars, b_chars_len = pad_char_sequences(batch_chars)
    if batch_labels is None:
        return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}
    else:
        b_puncs, _ = pad_sequences(batch_labels[0])
        b_ners, _ = pad_sequences(batch_labels[1])
        return {"words": b_words, "chars": b_chars, "puncs": b_puncs, "ners": b_ners, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size):
    batch_words, batch_chars, batch_puncs, batch_ners = [], [], [], []
    i = 0
    for record in dataset:
        batch_words.append(record["words"])
        batch_chars.append(record["chars"])
        batch_puncs.append(record["puncs"])
        batch_ners.append(record["ners"])
        if len(record["chars"]) == 0:
            print(i)
        i += 1
        if len(batch_words) == batch_size:
            yield process_batch_data(batch_words, batch_chars, [batch_puncs, batch_ners])
            batch_words, batch_chars, batch_puncs, batch_ners = [], [], [], []
    if len(batch_words) > 0:
        yield  process_batch_data(batch_words, batch_chars, [batch_puncs, batch_ners])

def batchnize_dataset(data, batch_size=None, shuffle=True):
    if type(data) == str:
        dataset = load_data(data)
    else:
        dataset = data
    if shuffle:
        random.shuffle(dataset)
    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset)):
            batches.append(batch)
#        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size):
            batches.append(batch)
        return batches


class BiLSTM_model:
    def __init__(self, config):
        self.cfg = config
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"],"log.txt"))

        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time - 1)
        self.seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.compat.v1.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.compat.v1.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.compat.v1.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.compat.v1.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(params=self.char_embeddings, ids=self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            self.word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.compat.v1.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.compat.v1.name_scope("BiLSTM"):
            with tf.compat.v1.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.compat.v1.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_= bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len, dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

#            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))

            self.logits = tf.compat.v1.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.seq_len)
        self.loss = tf.math.reduce_mean(input_tensor=tf.boolean_mask(tensor=losses, mask=mask))
#        losses = focal_loss(self.gamma, self.alpha)
#        self.loss = losses(self.labels, self.logits)
#        self.loss = tf.reduce_mean(self.loss)
        tf.compat.v1.summary.scalar("loss", self.loss)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.compat.v1.summary.merge_all()
        self.train_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=0.5, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        pred_logits = tf.cast(tf.argmax(input=self.logits, axis=-1), tf.int32)
        logits = self.sess.run(pred_logits, feed_dict=feed_dict)
        return logits

    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Valid micro average fscore: {}'.format(micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'EXCLAM', 'COLON', 'QMARK','SEMICOLON']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        # metrics = [y for x in labels for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0

        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro

class BiLSTM_Attention_model:
    def __init__(self, config):
        self.cfg = config
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"],"log.txt"))
        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time - 1)
        self.seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.compat.v1.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.compat.v1.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.compat.v1.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.compat.v1.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(params=self.char_embeddings, ids=self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.compat.v1.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.compat.v1.name_scope("BiLSTM"):
            with tf.compat.v1.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.compat.v1.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

            rnn_outs = tf.compat.v1.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))
            context = tf.transpose(a=outputs, perm=[1, 0, 2])
            p_context = tf.compat.v1.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
            p_context = tf.transpose(a=p_context, perm=[1, 0, 2])
            attn_cell = AttentionCell(self.cfg["num_units"], context, p_context)  # time major based
            attn_outs, _ = dynamic_rnn(attn_cell, context, sequence_length=self.seq_len, time_major=True,
                                       dtype=tf.float32)
            outputs = tf.transpose(a=attn_outs, perm=[1, 0, 2])
            print("Attention output shape: {}".format(outputs.get_shape().as_list()))
            self.logits = tf.compat.v1.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.seq_len)
        self.loss = tf.reduce_mean(input_tensor=tf.boolean_mask(tensor=losses, mask=mask))
#        losses = focal_loss(self.gamma,self.alpha)
#        self.loss = losses(self.labels, self.logits)
#        self.loss = tf.reduce_mean(self.loss)
        tf.compat.v1.summary.scalar("loss", self.loss)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.compat.v1.summary.merge_all()
        self.train_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        pred_logits = tf.cast(tf.argmax(input=self.logits, axis=-1), tf.int32)
        logits = self.sess.run(pred_logits, feed_dict=feed_dict)
        return logits
    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Valid micro average fscore: {}'.format(micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'QMARK']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0

        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro

class BiLSTM_CRF_model:
    def __init__(self, config):
        self.cfg = config
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"],"log.txt"))
        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time)
        self.seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.compat.v1.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.compat.v1.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.compat.v1.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.compat.v1.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(params=self.char_embeddings, ids=self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.compat.v1.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.compat.v1.name_scope("BiLSTM"):
            with tf.compat.v1.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.compat.v1.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

#            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))

            self.logits = tf.compat.v1.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
        crf_loss, self.trans_params = crf_log_likelihood(self.logits, self.labels, self.seq_len)
#        losses = focal_loss(self.gamma,self.alpha)
#        self.loss = losses(self.labels, self.logits)
        self.loss = tf.reduce_mean(input_tensor=-crf_loss)
        tf.compat.v1.summary.scalar("loss", self.loss)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.compat.v1.global_variables_initializer())


    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.compat.v1.summary.merge_all()
        self.train_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, trans_params, seq_len = self.sess.run([self.logits, self.trans_params, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, trans_params, seq_len)

    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Valid micro average fscore: {}'.format(micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'QMARK']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0
        
        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro


class Multitask_model:
    def __init__(self, config):
        self.cfg = config
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"],"log.txt"))
        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.punc_dict = dict_data["punc_dict"]
        self.ner_dict = dict_data["ner_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.punc_label_vocab_size = len(self.punc_dict)
        self.ner_label_vocab_size = len(self.ner_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.punc_labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="punc") # shape = (batch_size, max_time - 1)
        self.ner_labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="ner")
        self.seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.compat.v1.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.compat.v1.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.compat.v1.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.compat.v1.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(params=self.char_embeddings, ids=self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.compat.v1.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.compat.v1.name_scope("BiLSTM"):
            with tf.compat.v1.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.compat.v1.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

            rnn_outs = tf.compat.v1.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))
            context = tf.transpose(a=outputs, perm=[1, 0, 2])
            
            p_context = tf.compat.v1.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
            p_context = tf.transpose(a=p_context, perm=[1, 0, 2])
            p_attn_cell = Punc_AttentionCell(self.cfg["num_units"], context, p_context)  # time major based
            p_attn_outs, _ = dynamic_rnn(p_attn_cell, context, sequence_length=self.seq_len, time_major=True,
                                       dtype=tf.float32)
            p_outputs = tf.transpose(a=p_attn_outs, perm=[1, 0, 2])
            print("Attention output shape of punctuation: {}".format(p_outputs.get_shape().as_list()))

            n_context = tf.compat.v1.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
            n_context = tf.transpose(a=n_context, perm=[1, 0, 2])
            n_attn_cell = Ner_AttentionCell(self.cfg["num_units"], context, n_context)  # time major based
            n_attn_outs, _ = dynamic_rnn(n_attn_cell, context, sequence_length=self.seq_len, time_major=True,
                                       dtype=tf.float32)
            n_outputs = tf.transpose(a=n_attn_outs, perm=[1, 0, 2])
            print("Attention output shape of ner: {}".format(n_outputs.get_shape().as_list()))

            self.ner_logits = tf.compat.v1.layers.dense(n_outputs, units=self.ner_label_vocab_size, use_bias=True)
            self.ner_losses = tf.nn.softmax(self.ner_logits)
            print("Ner logits shape: {}".format(self.ner_logits.get_shape().as_list()))
            self.punc_logits = tf.compat.v1.layers.dense(p_outputs, units=self.punc_label_vocab_size, use_bias=True)
            self.punc_losses = tf.nn.softmax(self.punc_logits)
            print("Punc logits shape: {}".format(self.punc_logits.get_shape().as_list()))
            self.losses = tf.concat([self.punc_losses, self.ner_losses], -1)
            print("Name of losses: {}".format(self.losses.name))
            self.pred_punc_logits = tf.cast(tf.argmax(input=self.punc_losses, axis=-1), tf.int32)
            self.pred_ner_logits = tf.cast(tf.argmax(input=self.ner_losses, axis=-1), tf.int32)
            self.pred_logits = tf.concat([self.pred_punc_logits, self.pred_ner_logits], 0)
            self.ml_punc = tf.reduce_max(input_tensor=self.punc_losses, axis=-1)
            self.ml_ner = tf.reduce_max(input_tensor=self.ner_losses, axis=-1)
            self.pred_logits = tf.concat([self.pred_punc_logits, self.pred_ner_logits], 0)
            self.ml = tf.concat([self.ml_punc, self.ml_ner], 0)
        # Define loss and optimizer
        mask = tf.sequence_mask(self.seq_len)
        p_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.punc_logits, labels=self.punc_labels)
        self.punc_loss = tf.reduce_mean(input_tensor=tf.boolean_mask(tensor=p_losses, mask=mask))
        n_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ner_logits, labels=self.ner_labels)
        self.ner_loss = tf.reduce_mean(input_tensor=tf.boolean_mask(tensor=n_losses, mask=mask))
        self.loss = self.ner_loss + self.punc_loss
        tf.compat.v1.summary.scalar("loss", self.loss)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

        sess_config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
        sess_config.gpu_options.allow_growth = False
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            tf.io.write_graph(self.sess.graph, './', "model.pb", as_text=False)
            print("___________________ ", ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.compat.v1.summary.merge_all()
        self.train_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.compat.v1.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "puncs" in batch and "ners" in batch:
            feed_dict[self.punc_labels] = batch["puncs"]
            feed_dict[self.ner_labels] = batch["ners"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits = self.sess.run(self.pred_logits, feed_dict=feed_dict)
        loss = self.sess.run(self.ml, feed_dict=feed_dict)
        return np.array_split(logits, 2), np.array_split(loss, 2)
    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        n_best_f1, p_best_f1, no_imprv_epoch = -np.inf,-np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Valid micro average fscore of punctuation: {}'.format(micro_f_val[0]))
            self.logger.info('Valid micro average fscore of ner: {}'.format(micro_f_val[1]))
            p_cur_f1 = micro_f_val[0]
            n_cur_f1 = micro_f_val[1]
            if p_cur_f1 > p_best_f1 or n_cur_f1 > n_best_f1:
               no_imprv_epoch = 0
               p_best_f1 = p_cur_f1
               n_best_f1 = n_cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f} - {:04.2f}".format(epoch, p_best_f1, n_best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set, thres_hold = [1,1]):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test", thres_hold = thres_hold)
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro[0]))
        self.logger.info("\n{}\n".format(micro[1]))
    def test_sentence(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'QMARK']
        pp = [' ', '.',',','?']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]
        p_label, n_label = [], []
        print("Len dataset: ", len(dataset))
        for data in dataset:
            punc, ner = self._predict_op(data)
            print("Punc ___", punc)
            print("Ner ___", ner)
            for ppp in punc:
                for p in ppp:
                    p_label.append(pp[p])
            for nnn in ner:
                for n in nnn:
                    n_label.append(n)
        return p_label, n_label
    def evaluate_punct(self, dataset, name, thres_hold = [1,1]):
        PUNCTUATIONS = ['O','P', 'C', 'Q']
        NER = ['O', 'E']
        punc_preds, ner_preds = [], []
        punc_labels, ner_labels = [], []

        N_TP = P_TP = 0.0
        N_FP = P_FP = 0.0
        N_FN = P_FN = 0.0

        
        num_punc_class = len(PUNCTUATIONS)
        num_ner_class = len(NER)
#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts, loss = self._predict_op(data)
            punc_predicts, ner_predicts = predicts
            punc_loss, ner_loss = loss
            for i, puncs in enumerate(punc_loss):
                for j, punc in enumerate(puncs):
                    if punc < thres_hold[0]:
                        punc_predicts[i][j] = 0
            for i, ners in enumerate(ner_loss):
                for j, ner in enumerate(ners):
                    if ner < thres_hold[1]:
                        ner_predicts[i][j] = 0
            for punc_pred, ner_pred, punc_tag, ner_tag, seq_len in zip(punc_predicts, ner_predicts, data["puncs"], data["ners"], data["seq_len"]):
                punc_preds.append(punc_pred[:seq_len])
                ner_preds.append(ner_pred[:seq_len])
#                print(preds)
                punc_labels.append(punc_tag[:seq_len])
                ner_labels.append(ner_tag[:seq_len])
                for i in range(len(punc_pred)):
                    for l in range(1,7):
                        if (punc_pred[i] == punc_tag[i]) and (punc_tag[i] == l):
                            P_TP += 1
                        elif (punc_pred[i] != punc_tag[i]) and (punc_tag[i] == l):
                            P_FN += 1
                        elif (punc_pred[i] != punc_tag[i]) and (punc_pred[i] == l):
                            P_FP += 1
                for i in range(len(ner_pred)):
                    for l in range(1,7):
                        if (ner_pred[i] == ner_tag[i]) and (ner_tag[i] == l):
                            N_TP += 1
                        elif (ner_pred[i] != ner_tag[i]) and (ner_tag[i] == l):
                            N_FN += 1
                        elif (ner_pred[i] != ner_tag[i]) and (ner_pred[i] == l):
                            N_FP += 1
        punc_labels = [y for x in punc_labels for y in x]
        punc_preds = [y for x in punc_preds for y in x]

        ner_labels = [y for x in ner_labels for y in x]
        ner_preds = [y for x in ner_preds for y in x]



        punc_precision = precision_score(punc_labels, punc_preds, average=None)
        punc_recall = recall_score(punc_labels, punc_preds, average=None)
        punc_f_score = f1_score(punc_labels, punc_preds, average=None)

        ner_precision = precision_score(ner_labels, ner_preds, average=None)
        ner_recall = recall_score(ner_labels, ner_preds, average=None)
        ner_f_score = f1_score(ner_labels, ner_preds, average=None)



        if (P_TP + P_FN) != 0:
            p_micro_r = P_TP / (P_TP + P_FN)
        else:
            p_micro_r = 0
        ###################
        if (P_TP + P_FP) != 0:
            p_micro_p = P_TP / (P_TP + P_FP)
        else:
            p_micro_p = 0
        ################
        if (p_micro_r + p_micro_p) > 0:
            p_micro_f = 2*p_micro_r * p_micro_p / (p_micro_r + p_micro_p)
        else:
            p_micro_f = 0.0



        if (N_TP + N_FN) != 0:
            n_micro_r = N_TP / (N_TP + N_FN)
        else:
            n_micro_r = 0
        ###################
        if (N_TP + N_FP) != 0:
            n_micro_p = N_TP / (N_TP + N_FP)
        else:
            n_micro_p = 0
        ################
        if (n_micro_r + n_micro_p) > 0:
            n_micro_f = 2*n_micro_r * n_micro_p / (n_micro_r + n_micro_p)
        else:
            n_micro_f = 0.0

        p_micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*p_micro_p) + '%\n\tRecall: ' + str(100*p_micro_r) + ' %\n\t F_1 score: '  + str(100*p_micro_f) + ' %\n'
        n_micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*n_micro_p) + '%\n\tRecall: ' + str(100*n_micro_r) + ' %\n\t F_1 score: '  + str(100*n_micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_punc_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*punc_precision[i]),
                                                            "{:.4f}".format(100*punc_recall[i]),
                                                            "{:.4f}".format(100*punc_f_score[i]))
        out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(NER[1], "{:.4f}".format(100*ner_precision[1]),
                                                            "{:.4f}".format(100*ner_recall[1]),
                                                            "{:.4f}".format(100*ner_f_score[1]))

        return [p_micro_f, n_micro_f], out_str, [p_micro, n_micro]
