#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bidirectional_irnn_layer.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype relu(const Dtype x) {
  return (x >= (Dtype)0)? x : 0;
}

template <typename Dtype>
__global__ void CondAdd(const int nthreads, int t, const Dtype* add_vec, Dtype* data, bool cond) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype clip_t = cond;
    data[index] += clip_t * add_vec[index];
  }
}

template <typename Dtype>
__global__ void ActivationForward(const int nthreads, const Dtype* pre_gate, Dtype* gate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    gate[index] = relu(pre_gate[index]);
  }
}

template <typename Dtype>
__global__ void ActivationBackward(const int nthreads, const Dtype* pre_gate, const Dtype* dh_t, Dtype* pre_gate_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    pre_gate_diff[index] = dh_t[index] * ( (pre_gate[index] > Dtype(0) )? 1 : 0);
  }
}

template <typename Dtype>
void BidirectionalIRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* bias = this->blobs_[2]->gpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_gpu_data();

  caffe_gpu_set(h_0_.count(), Dtype(0.), h_0_.mutable_gpu_data());

  // Compute input to hidden forward propagation
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, T_*N_, H_, I_, Dtype(1.),
      bottom_data, weight_i, Dtype(0.), pre_gate_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, H_, 1, Dtype(1.),
      bias_multiplier_.gpu_data(), bias, Dtype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + top_.offset(t);
    Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();

    caffe_gpu_gemm(CblasNoTrans, CblasTrans, N_, H_, H_, Dtype(1.),
        h_t_1, weight_h, Dtype(0.), h_to_gate_.mutable_gpu_data());
    CondAdd<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, t, h_to_gate_.gpu_data(), pre_gate_t,t>0);
    CUDA_POST_KERNEL_CHECK;
    ActivationForward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, pre_gate_t, h_t);
    CUDA_POST_KERNEL_CHECK;
  }

  caffe_copy(N_*H_, top_data + top_.offset(T_-1), top[0]->mutable_gpu_data());

  //backward direction irnn
  top_data = backdirect_top_.mutable_gpu_data();
  weight_i = this->blobs_[3]->gpu_data();
  weight_h = this->blobs_[4]->gpu_data();
  bias = this->blobs_[5]->gpu_data();
  pre_gate_data = backdirect_pre_gate_.mutable_gpu_data();

  caffe_gpu_set(h_0_.count(), Dtype(0.), h_0_.mutable_gpu_data());

  // Compute input to hidden forward propagation
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, T_*N_, H_, I_, Dtype(1.),
      bottom_data, weight_i, Dtype(0.), pre_gate_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, H_, 1, Dtype(1.),
      bias_multiplier_.gpu_data(), bias, Dtype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = T_-1; t >=0; --t) {
    Dtype* h_t = top_data + backdirect_top_.offset(t);
    Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    const Dtype* h_t_1 = (t!=(T_-1)) ? (h_t + top_.offset(1)) : h_0_.gpu_data();

    caffe_gpu_gemm(CblasNoTrans, CblasTrans, N_, H_, H_, Dtype(1.),
        h_t_1, weight_h, Dtype(0.), h_to_gate_.mutable_gpu_data());
    CondAdd<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, t, h_to_gate_.gpu_data(), pre_gate_t,t!=(T_-1));
    CUDA_POST_KERNEL_CHECK;
    ActivationForward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, pre_gate_t, h_t);
    CUDA_POST_KERNEL_CHECK;
  }

  caffe_copy(N_*H_, top_data, top[1]->mutable_gpu_data());
}

template <typename Dtype>
void BidirectionalIRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.gpu_data();
  Dtype* top_diff = top_.mutable_gpu_diff();
  caffe_gpu_set(top_.count(), Dtype(0), top_diff);
  caffe_copy(N_*H_, top[0]->mutable_gpu_diff(), top_diff + top_.offset(T_-1));

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();

  const Dtype* pre_gate_data = pre_gate_.gpu_data();
  Dtype* pre_gate_diff = pre_gate_.mutable_gpu_diff();

  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + top_.offset(t);
    Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
    Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_gpu_diff();
    const Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);

    ActivationBackward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, pre_gate_t, dh_t, pre_gate_diff_t);
    CUDA_POST_KERNEL_CHECK;

    // Backprop errors to the previous time step
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, H_,
        Dtype(1.), pre_gate_diff_t, weight_h, Dtype(0.), h_to_h_.mutable_gpu_data());
    CondAdd<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, t, h_to_h_.gpu_data(), dh_t_1,t>0);
  }

  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, I_, T_*N_, Dtype(1.),
        pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, H_, (T_-1)*N_, Dtype(1.),
        pre_gate_diff + pre_gate_.offset(1), top_data,
        Dtype(1.), this->blobs_[1]->mutable_gpu_diff());

    // Add Gradient from previous time-step
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, H_, 1, Dtype(1.),
        pre_gate_diff, h_0_.gpu_data(),
        Dtype(1.), this->blobs_[1]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[2]) {
    // Gradient w.r.t. bias
    caffe_gpu_gemv(CblasTrans, T_*N_, H_, Dtype(1.), pre_gate_diff,
        bias_multiplier_.gpu_data(), Dtype(1.),
        this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, H_, Dtype(1.),
        pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_gpu_diff());
  }


  //backward directional irnn
  top_data = backdirect_top_.gpu_data();
  top_diff = backdirect_top_.mutable_gpu_diff();
  caffe_gpu_set(top_.count(), Dtype(0), top_diff);
  caffe_copy(N_*H_, top[1]->mutable_gpu_diff(), top_diff);

  weight_i = this->blobs_[3]->gpu_data();
  weight_h = this->blobs_[4]->gpu_data();

  pre_gate_data = backdirect_pre_gate_.gpu_data();
  pre_gate_diff = backdirect_pre_gate_.mutable_gpu_diff();

  for (int t = 0; t < T_; ++t) {
    Dtype* dh_t = top_diff + backdirect_top_.offset(t);
    Dtype* pre_gate_diff_t = pre_gate_diff + backdirect_pre_gate_.offset(t);
    Dtype* dh_t_1 = (t!=(T_-1)) ? top_diff + backdirect_top_.offset(t+1) : h_0_.mutable_gpu_diff();
    const Dtype* pre_gate_t = pre_gate_data + backdirect_pre_gate_.offset(t);

    ActivationBackward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, pre_gate_t, dh_t, pre_gate_diff_t);
    CUDA_POST_KERNEL_CHECK;

    // Backprop errors to the previous time step
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, H_,
        Dtype(1.), pre_gate_diff_t, weight_h, Dtype(0.), h_to_h_.mutable_gpu_data());
    CondAdd<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, t, h_to_h_.gpu_data(), dh_t_1,t!=(T_-1));
  }

  if (this->param_propagate_down_[3]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, I_, T_*N_, Dtype(1.),
        pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[3]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[4]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, H_, (T_-1)*N_, Dtype(1.),
        pre_gate_diff, top_data + backdirect_top_.offset(1),
        Dtype(1.), this->blobs_[4]->mutable_gpu_diff());

    /*// Add Gradient from previous time-step
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, H_, 1, Dtype(1.),
        pre_gate_diff, h_0_.gpu_data(),
        Dtype(1.), this->blobs_[1]->mutable_gpu_diff());*/
  }
  if (this->param_propagate_down_[5]) {
    // Gradient w.r.t. bias
    caffe_gpu_gemv(CblasTrans, T_*N_, H_, Dtype(1.), pre_gate_diff,
        bias_multiplier_.gpu_data(), Dtype(1.),
        this->blobs_[5]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, H_, Dtype(1.),
        pre_gate_diff, weight_i, Dtype(1.), bottom[0]->mutable_gpu_diff());
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BidirectionalIRNNLayer);

}  // namespace caffe
