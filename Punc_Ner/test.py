from preprocessing import process_data
import preprocessing
import os
import time
from model_without_focal_loss import Multitask_model, batchnize_dataset
# dataset path
raw_path = 'dataset/Multitask_data'
save_path = "dataset/Encoded_multitask_large_data"
# embedding path
Word2vec_path = "embeddings"

char_lowercase = True
# dataset for train, validate and test
vocab = "dataset/Encoded_multitask_large_data/vocab.json"
train_set = "dataset/Encoded_multitask_large_data/train.json"
dev_set = "dataset/Encoded_multitask_large_data/dev.json"
test_set = "dataset/Encoded_multitask_large_data/test.json"
infer_test = "dataset/Encoded_multitask_large_data/a.json"
word_embedding = "dataset/Encoded_multitask_large_data/word_emb.npz"
# network parameters
num_units = 300
emb_dim = 300
char_emb_dim = 52
filter_sizes = [25, 25]
channel_sizes = [5, 5]
# training parameters
lr = 0.001
lr_decay = 0.05
minimal_lr = 1e-5
keep_prob = 0.5
batch_size = 32
epochs = 30
max_to_keep = 1
no_imprv_tolerance = 20
checkpoint_path = "checkpoint_multitask_large/"
summary_path = "checkpoint_multitask_large/summary/"
model_name = "model"

config = {"raw_path": raw_path,\
          "save_path": save_path,\
          "Word2vec_path":Word2vec_path,\
          "char_lowercase": char_lowercase,\
          "vocab": vocab,\
          "train_set": train_set,\
          "dev_set": dev_set,\
          "test_set": test_set,\
          "infer_test": infer_test,\
          "word_embedding": word_embedding,\
          "num_units": num_units,\
          "emb_dim": emb_dim,\
          "char_emb_dim": char_emb_dim,\
          "filter_sizes": filter_sizes,\
          "channel_sizes": channel_sizes,\
          "lr": lr,\
          "lr_decay": lr_decay,\
          "minimal_lr": minimal_lr,\
          "keep_prob": keep_prob,\
          "batch_size": batch_size,\
          "epochs": epochs,\
          "max_to_keep": max_to_keep,\
          "no_imprv_tolerance": no_imprv_tolerance,\
          "checkpoint_path": checkpoint_path,\
          "summary_path": summary_path,\
          "model_name": model_name}

import tensorflow as tf
tf.compat.v1.reset_default_graph()
print("Build models...")
model = Multitask_model(config)
model.restore_last_session(checkpoint_path)

test_set = batchnize_dataset(config["test_set"], batch_size=100, shuffle=False)
for i in range(4,10):
    punc_thres = i/10
    print("Run test with punc_thres = {}".format(punc_thres))
    model.test(test_set, thres_hold = [punc_thres, 0.0])
print("--------------------------------------------------------------------------------------------")
for i in range(4,10):
    ner_thres = i/10
    print("Run test with ner_thres = {}".format(ner_thres))
    model.test(test_set, thres_hold = [0.0, ner_thres])

#sen = "Ông chu ngọc anh được bầu làm chủ tịch ủy ban nhân dân thành phố hà nội sau khi ông nguyễn đức chung bị khởi tố tạm giam và bị bãi chức"
#sen = sen.lower()
#infer_test = preprocessing.process_data_infer(config, sen)
#print("test set: ==============" ,infer_test)
#preprocessing.write_json(os.path.join(config["save_path"], "a.json"), infer_test)
#infer_test = batchnize_dataset(config["infer_test"], batch_size=1000, shuffle=False)
## run the session
#ss = sen.lstrip().rstrip()
#w = ss.split(" ")
#start = time.time()
#p, n = model.test_sentence(infer_test, "test")
#end = time.time()
#print("Time predict:", end -start)
#out =''
#for i in range(0,len(w)):
#    if n[i] == 1:
#        w[i] = w[i].title()
#    out += w[i] + p[i]
#
#print("result: ", out)
#
