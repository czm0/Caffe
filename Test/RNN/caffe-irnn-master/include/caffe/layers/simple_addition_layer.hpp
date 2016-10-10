#ifndef CAFFE_SIMPLE_ADDITION_LAYER_HPP_
#define CAFFE_SIMPLE_ADDITION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Implementation of simple addition.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SimpleAdditionLayer : public Layer<Dtype> {
 public:
  explicit SimpleAdditionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SimpleAddition"; }

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
};

}  // namespace caffe

#endif  // CAFFE_SIMPLE_ADDITION_LAYER_HPP_
