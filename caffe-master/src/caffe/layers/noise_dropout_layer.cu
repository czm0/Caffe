#include<vector>
#include "caffe/layers/noise_dropout_layer.hpp"


namespace caffe
{
	template<typename Dtype>
	__global__ void NoiseDropoutForward(const int n, const unsigned int threshold, const unsigned int* mask, const Dtype* bottom_data, Dtype* top_data)
	{
		CUDA_KERNEL_LOOP(index, n) {
			top_data[index] = bottom_data[index] * (mask[index] > threshold);
		}
	}

	template<typename Dtype>
	void NoiseDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		unsigned int* mask =
			static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
		const int count = bottom[0]->count();
		caffe_gpu_rng_uniform(count, mask);
		NoiseDropoutForward<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (count, uint_thres_, mask, bottom_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NoiseDropoutLayer);
}