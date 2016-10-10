#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/simple_addition_layer.hpp"

namespace caffe {

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, N_*I_, T_, Dtype(1.),
        bias_multiplier_.cpu_data(), bottom_data, Dtype(0.), top_data);
}

template <typename Dtype>
void SimpleAdditionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    for( int t=0; t<T_; t++){
      caffe_copy(N_*H_, top_diff, bottom_diff + t*N_*H_);
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(SimpleAdditionLayer);

}  // namespace caffe
