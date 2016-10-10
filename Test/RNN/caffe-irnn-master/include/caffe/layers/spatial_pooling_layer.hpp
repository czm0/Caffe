#ifndef CAFFE_SPATIAL_POOLING_LAYER_HPP_
#define CAFFE_SPATIAL_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief feature encoding using spatial pooling.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SpatialPoolingLayer : public Layer<Dtype> {
 public:
  explicit SpatialPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_, kernel_t_;
  int height_, width_, duration_;
  int num_h_, num_w_, num_t_;
  Blob<int> max_idx_;
  int num_traj_, hidden_size_;
};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_POOLING_LAYER_HPP_
