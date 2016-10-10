#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolingForward(const int hidden_size_, int n, int index_pos, const Dtype* feature_data, Dtype* top_data, int* mask) {
  CUDA_KERNEL_LOOP(len, hidden_size_) {
    if ( feature_data[n*hidden_size_+len] > top_data[index_pos*hidden_size_+len] ){
        top_data[index_pos*hidden_size_+len] = feature_data[n*hidden_size_+len];
        mask[index_pos*hidden_size_+len] = n*hidden_size_+len;
    }
  }
}

template <typename Dtype>
__global__ void ProcessEdge(const int hidden_size_, int i, Dtype* top_data) {
  CUDA_KERNEL_LOOP(len, hidden_size_) {
    int index_top = i*hidden_size_+len;
    if(top_data[index_top] < Dtype(-1e5)){
       top_data[index_top]=Dtype(0);
    }
  }
}

template <typename Dtype>
__global__ void Backfill(const int hidden_size_, int i, const Dtype* top_diff, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(len, hidden_size_) {
    int index_top = i*hidden_size_+len;
    if(mask[index_top] != -1){
        bottom_diff[ mask[index_top] ] = top_diff[index_top];
    }
  }
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  num_traj_=bottom[0]->num();
  const Dtype* position_data = bottom[1]->cpu_data();
  height_ = position_data[num_traj_*3];
  width_ =  position_data[num_traj_*3+1];
  duration_= position_data[num_traj_*3+2];
  kernel_h_ = height_ / num_h_;
  kernel_w_ = width_ / num_w_;
  kernel_t_ = duration_ / num_t_ + ((duration_ % num_t_)?1:0);
  const Dtype* feature_data = bottom[0]->gpu_data();
  //const Dtype* position_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();

  int* mask = max_idx_.mutable_gpu_data();
  caffe_gpu_set(top_count, -1, mask);

  caffe_gpu_set(top_count, Dtype(-1e10), top_data);
  // The main loop
  int index_h, index_w, index_t,index_pos;
  for (int n = 0; n < num_traj_; ++n) {
    index_h = position_data[n*3]/kernel_h_;
    index_w = position_data[n*3+1]/kernel_w_;
    index_t = position_data[n*3+2]/kernel_t_;
    index_pos = index_h + num_h_*(index_w + num_w_*index_t);

    MaxPoolingForward<Dtype><<<CAFFE_GET_BLOCKS(hidden_size_), CAFFE_CUDA_NUM_THREADS>>>(hidden_size_, n, index_pos, feature_data, top_data, mask);
  }
  int sum_pos=num_h_*num_w_*num_t_;
  for ( int i =0; i<sum_pos; i++){
      ProcessEdge<Dtype><<<CAFFE_GET_BLOCKS(hidden_size_), CAFFE_CUDA_NUM_THREADS>>>(hidden_size_, i,top_data);
  }

}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
   if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

  int sum_pos=num_h_*num_w_*num_t_;
  const int* mask = max_idx_.gpu_data();
  for ( int i =0; i<sum_pos; i++){
    Backfill<Dtype><<<CAFFE_GET_BLOCKS(hidden_size_), CAFFE_CUDA_NUM_THREADS>>>(hidden_size_, i,top_diff, mask, bottom_diff);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialPoolingLayer);

}  // namespace caffe
