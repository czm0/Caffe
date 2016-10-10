#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/spatial_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SpatialPoolingParameter pool_param = this->layer_param_.spatial_pooling_param();

  hidden_size_=bottom[0]->channels();
  num_h_=pool_param.num_h();
  num_w_=pool_param.num_w();
  num_t_=pool_param.num_t();
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1,num_h_*num_w_*num_t_, hidden_size_,1);

  // Using max pooling, we will initialize the vector index part.
  max_idx_.Reshape(num_h_*num_w_*num_t_, hidden_size_,1,1);
}


template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_traj_=bottom[0]->num();
  const Dtype* position_data = bottom[1]->cpu_data();
  height_ = position_data[num_traj_*3];
  width_ =  position_data[num_traj_*3+1];
  duration_= position_data[num_traj_*3+2];
  kernel_h_ = height_ / num_h_;
  kernel_w_ = width_ / num_w_;
  kernel_t_ = duration_ / num_t_ + ((duration_ % num_t_)?1:0);
  const Dtype* feature_data = bottom[0]->cpu_data();
  //const Dtype* position_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  int* mask = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, mask);

  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // The main loop
  int index_h, index_w, index_t,index_pos;
  for (int n = 0; n < num_traj_; ++n) {
    index_h = position_data[n*3]/kernel_h_;
    index_w = position_data[n*3+1]/kernel_w_;
    index_t = position_data[n*3+2]/kernel_t_;
    index_pos = index_h + num_h_*(index_w + num_w_*index_t);

    for (int len = 0; len < hidden_size_; len++){
        if ( feature_data[n*hidden_size_+len] > top_data[index_pos*hidden_size_+len] ){
            top_data[index_pos*hidden_size_+len] = feature_data[n*hidden_size_+len];
            mask[index_pos*hidden_size_+len] = n*hidden_size_+len;
        }
    }
  }
  int sum_pos=num_h_*num_w_*num_t_, index_top;
  for ( int i =0; i<sum_pos; i++){
    for (int len = 0; len < hidden_size_; len++){
        index_top = i*hidden_size_+len;
        if(top_data[index_top] < Dtype(-FLT_MAX)/2){
           top_data[index_top]=0;
        }
    }
  }
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  int sum_pos=num_h_*num_w_*num_t_, index_top;
  const int* mask = max_idx_.cpu_data();
  for ( int i =0; i<sum_pos; i++){
    for (int len = 0; len < hidden_size_; len++){
        index_top = i*hidden_size_+len;
        if(mask[index_top] != -1){
            bottom_diff[ mask[index_top] ] = top_diff[index_top];
        }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialPoolingLayer);
#endif

INSTANTIATE_CLASS(SpatialPoolingLayer);
REGISTER_LAYER_CLASS(SpatialPooling);

}  // namespace caffe
