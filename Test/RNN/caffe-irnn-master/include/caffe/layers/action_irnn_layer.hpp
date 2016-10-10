#ifndef CAFFE_ACTION_IRNN_LAYER_HPP_
#define CAFFE_ACTION_IRNN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Implementation of Identity Recurrent Neural Networks(IRNN).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ActionIRNNLayer : public Layer<Dtype> {
 public:
  explicit ActionIRNNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ActionIRNN"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int N_; // batch size

  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> top_;       // output values
  Blob<Dtype> pre_gate_;  // gate values before nonlinearity

  Blob<Dtype> h_0_; // previous hidden activation value
  Blob<Dtype> h_T_; // next hidden activation value

  // intermediate values
  Blob<Dtype> h_to_gate_;
  Blob<Dtype> h_to_h_;
};

}  // namespace caffe

#endif  // CAFFE_ACTION_IRNN_LAYER_HPP_
