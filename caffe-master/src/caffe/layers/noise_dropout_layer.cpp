#include"caffe/layers/noise_dropout_layer.hpp"

namespace caffe
{
	template<typename Dtype>
	void NoiseDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
		threshold_ = this->layer_param_.noise_dropout_param().dropout_ratio();
		DCHECK(threshold_ > 0.);
		DCHECK(threshold_ < 1.);
		this->param_propagate_down_.resize(this->blobs_.size(), false);	//º∆À„diff

		uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
	}
	template<typename Dtype>
	void NoiseDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		NeuronLayer<Dtype>::Reshape(bottom,top);
		rand_vec_.Reshape(bottom[0]->shape());
	}
	template<typename Dtype>
	void NoiseDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		unsigned int* mask = rand_vec_.mutable_cpu_data();
		const int count = bottom[0]->count();
		caffe_rng_bernoulli(count, 1. - threshold_, mask);
		for (int i = 0; i < count; ++i) {
			top_data[i] = bottom_data[i] * mask[i];
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(NoiseDropoutLayer);
#endif
	INSTANTIATE_CLASS(NoiseDropoutLayer);
	REGISTER_LAYER_CLASS(NoiseDropout);
}