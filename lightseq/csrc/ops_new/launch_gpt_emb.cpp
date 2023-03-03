#include "launch_gpt_emb.h"

namespace lightseq {

template <typename T>
std::tuple<Variable*, Variable*> LaunchGptEmbOp<T>::operator()(
    Variable* inp_tokens, Variable* token_emb, Variable* pos_emb) {
  set_parents({inp_tokens, token_emb, pos_emb});

  size_t max_size = _max_batch_tokens * _hidden_dim;

  _result = new Variable("LaunchGptEmbOp_out", _max_batch_tokens * _hidden_dim,
                         g_dtype<T>());
  _result_seq_len = new Variable("pad_mask", _max_batch_tokens, g_dtype<int>());
  this->set_children({_result, _result_seq_len});
  return std::make_tuple(_result, _result_seq_len);
}

template <typename T>
void LaunchGptEmbOp<T>::forward() {
  int* inp_tokens = (int*)parent(0)->value();
  const T* token_emb = (const T*)parent(1)->value();
  const T* pos_emb = (const T*)parent(2)->value();

  T* output_ptr = (T*)child(0)->value();
  int* seq_len_ptr = (int*)child(1)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  cuda::ker_gpt_embedding_launcher<T>(
      _batch_size, _seq_len, _hidden_dim, _stream, token_emb, pos_emb,
      inp_tokens, output_ptr, seq_len_ptr, _pad_id, _offset);
#endif
}

template class LaunchGptEmbOp<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchGptEmbOp<__half>;
#endif
}  // namespace lightseq
