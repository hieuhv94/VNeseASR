
/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tensorflow/c/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct feed_dict_t {
    int32_t* words;
    int32_t* chars;
    int32_t max_word_len;
    int32_t seq_len;
    char is_train;
    float drop_rate;
} feed_dict_t;

typedef struct model_t {
  TF_Graph* graph;
  TF_Session* session;
  TF_Status* status;

  TF_Output words, chars, seq_len, is_train, drop_rate, output;

  TF_Operation *restore_op;
  TF_Output checkpoint_file;
} model_t;

int ModelCreate(model_t* model, const char* graph_def_filename);
void ModelDestroy(model_t* model);
int ModelPredict(model_t* model, feed_dict_t feed_dict,  float* predicts);
int ModelCheckpoint(model_t* model, const char* checkpoint_prefix);

int Okay(TF_Status* status);
TF_Buffer* ReadFile(const char* filename);
TF_Tensor* ScalarStringTensor(const char* data, TF_Status* status);

#ifdef __cplusplus
} /* end of "extern C" block */
#endif