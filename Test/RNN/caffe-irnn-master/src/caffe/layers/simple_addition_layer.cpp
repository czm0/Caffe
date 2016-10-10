#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/simple_addition_layer.hpp"

namespace caffe {

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  I_ = bottom[0]->width();
  H_ = I_;
}

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  T_ = this->layer_param_.simple_addition_param().trajectory_len(); // trajectory_len
  N_ = bottom[0]->height();
  vector<int> original_top_shape;
  original_top_shape.push_back(N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  // Set up the bias multiplier
  vector<int> multiplier_shape(1, T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1),
    bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, N_*I_, T_, Dtype(1.),
      bias_multiplier_.cpu_data(), bottom_data, Dtype(0.), top_data);
}

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for( int t=0; t<T_; t++){
    caffe_copy(N_*H_, top_diff, bottom_diff + t*N_*H_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimpleAdditionLayer);
#endif

INSTANTIATE_CLASS(SimpleAdditionLayer);
REGISTER_LAYER_CLASS(SimpleAddition);

}  // namespace caffe
