/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "tf.h"

int ModelCreate(model_t* model, const char* graph_def_filename) {
  model->status = TF_NewStatus();
  model->graph = TF_NewGraph();

  {
    // Create the session.
    TF_SessionOptions* opts = TF_NewSessionOptions();
    model->session = TF_NewSession(model->graph, opts, model->status);
    TF_DeleteSessionOptions(opts);
    if (!Okay(model->status)) return 0;
  }

  TF_Graph* g = model->graph;

  {
    // Import the graph.
    TF_Buffer* graph_def = ReadFile(graph_def_filename);
    if (graph_def == NULL) return 0;
    printf("Read GraphDef of %zu bytes\n", graph_def->length);
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(g, graph_def, opts, model->status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);
    if (!Okay(model->status)) return 0;
  }
  // Handles to the interesting operations in the graph.
  model->words.oper = TF_GraphOperationByName(g, "words");
  model->words.index = 0;
  model->chars.oper = TF_GraphOperationByName(g, "chars");
  model->chars.index = 0;
  model->drop_rate.oper = TF_GraphOperationByName(g, "dropout_rate");
  model->drop_rate.index = 0;
  model->is_train.oper = TF_GraphOperationByName(g, "is_train");
  model->is_train.index = 0;
  model->seq_len.oper = TF_GraphOperationByName(g, "seq_len");
  model->seq_len.index = 0;

  model->output.oper = TF_GraphOperationByName(g, "BiLSTM/concat_1");
  model->output.index = 0;
  
  model->restore_op = TF_GraphOperationByName(g, "save/restore_all");

  model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
  model->checkpoint_file.index = 0;
  return 1;
}

void ModelDestroy(model_t* model) {
  TF_DeleteSession(model->session, model->status);
  Okay(model->status);
  TF_DeleteGraph(model->graph);
  TF_DeleteStatus(model->status);
}

int ModelCheckpoint(model_t* model, const char* checkpoint_prefix) {
  TF_Tensor* t = ScalarStringTensor(checkpoint_prefix, model->status);
  if (!Okay(model->status)) {
    TF_DeleteTensor(t);
    return 0;
  }
  TF_Output inputs[1] = {model->checkpoint_file};
  TF_Tensor* input_values[1] = {t};
  const TF_Operation* op[1] = {model->restore_op};
  TF_SessionRun(model->session, NULL, inputs, input_values, 1,
                /* No outputs */
                NULL, NULL, 0,
                /* The operation */
                op, 1, NULL, model->status);
  TF_DeleteTensor(t);
  return Okay(model->status);
}

int ModelPredict(model_t* model, feed_dict_t feed_dict, float* predicts) {
  // batch consists of 1x1 matrices.
  const int64_t dim1[2] = {1, feed_dict.seq_len};
  const int64_t dim2[1] = {1};
  const int64_t dim3[3] = {1, feed_dict.seq_len, feed_dict.max_word_len};
  const int64_t dim4[1] = {1};
  const int64_t dim5[1] = {1};

  TF_Tensor* input_values[5];
  input_values[0] = TF_AllocateTensor(TF_INT32, dim1, 2, feed_dict.seq_len * sizeof(int32_t));
  memcpy(TF_TensorData(input_values[0]), feed_dict.words, feed_dict.seq_len * sizeof(int32_t));
  input_values[1] = TF_AllocateTensor(TF_INT32, dim2, 1, 4);
  memcpy(TF_TensorData(input_values[1]), &feed_dict.seq_len, 4);
  input_values[2] = TF_AllocateTensor(TF_INT32, dim3, 3, feed_dict.seq_len * feed_dict.max_word_len * sizeof(int32_t));
  memcpy(TF_TensorData(input_values[2]), feed_dict.chars, feed_dict.seq_len * feed_dict.max_word_len * sizeof(int32_t));
  input_values[3] = TF_AllocateTensor(TF_BOOL, dim4, 0, 1);
  memcpy(TF_TensorData(input_values[3]), &feed_dict.is_train, 1);
  input_values[4] = TF_AllocateTensor(TF_FLOAT, dim5, 0, 4);
  memcpy(TF_TensorData(input_values[4]), &feed_dict.drop_rate, 4);
  TF_Output inputs[5] = {model->words, model->seq_len, model->chars, model->is_train, model->drop_rate};
  TF_Output outputs[1] = {model->output};
  TF_Tensor* output_values[1] = {NULL};
  TF_SessionRun(model->session, NULL, inputs, input_values, 5, outputs,
                output_values, 1,
                /* No target operations to run */
                NULL, 0, NULL, model->status);
  for (int i =0; i< 5; i++)
    TF_DeleteTensor(input_values[i]);
  if (!Okay(model->status)) return 0;
  
  size_t output_size = TF_TensorByteSize(output_values[0]);
  if (TF_TensorByteSize(output_values[0]) != 6 * feed_dict.seq_len * sizeof(float)) {
    fprintf(stderr,
            "ERROR: Expected predictions tensor to have %zu bytes, has %zu\n",
            4 * feed_dict.seq_len * sizeof(float), TF_TensorByteSize(output_values[0]));
    TF_DeleteTensor(output_values[0]);
    return 0;
  }
  if (!predicts)
    predicts = (float*)malloc(output_size);
  memcpy(predicts, TF_TensorData(output_values[0]), output_size);
  TF_DeleteTensor(output_values[0]);
  return 1;
}

int Okay(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: %s\n", TF_Message(status));
    return 0;
  }
  return 1;
}

TF_Buffer* ReadFile(const char* filename) {
  int fd = open(filename, 0);
  if (fd < 0) {
    perror("failed to open file: ");
    return NULL;
  }
  struct stat stat;
  if (fstat(fd, &stat) != 0) {
    perror("failed to read file: ");
    return NULL;
  }
  char* data = (char*)malloc(stat.st_size);
  ssize_t nread = read(fd, data, stat.st_size);
  if (nread < 0) {
    perror("failed to read file: ");
    free(data);
    return NULL;
  }
  if (nread != stat.st_size) {
    fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
            stat.st_size);
    free(data);
    return NULL;
  }
  TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
  free(data);
  return ret;
}

TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
  size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
  TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
  void* data = TF_TensorData(t);
  memset(data, 0, 8);  // 8-byte offset of first string.
  TF_StringEncode(str, strlen(str), data + 8, nbytes - 8, status);
  return t;
}
