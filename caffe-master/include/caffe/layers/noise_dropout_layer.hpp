#pragma once
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include <vector>
#include "caffe/util/math_functions.hpp"
namespace caffe
{
	template <typename Dtype>
	class NoiseDropoutLayer : public NeuronLayer < Dtype > {
	public:

		explicit NoiseDropoutLayer(const LayerParameter& param): NeuronLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "NoiseDropout"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
		void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

		Blob<unsigned int> rand_vec_;
		Dtype threshold_;
		unsigned int uint_thres_;
	};
}