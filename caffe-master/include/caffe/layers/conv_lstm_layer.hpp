#pragma once
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include <vector>
namespace caffe
{
	template<typename Dtype>
	class ConvolutionLSTMLayer : public Layer<Dtype>
	{
	public:
		ConvolutionLSTMLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "ConvolutionLstm"; }
		virtual inline int ExactNumBottomBlobs() const {return 1;}
		virtual inline int ExactNumTopBlobs() const {return 1;}
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		void compute_output_shape();
		inline int input_shape(int i) {
			return (*bottom_shape_)[1 + i];
		}
		void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* output,bool isWX);
		void forward_cpu_bias(Dtype* output, const Dtype* bias);

		void weight_cpu_gemm(const Dtype* data, const Dtype* gate_diff, Dtype* weight_diff, bool isWX);
		void backward_cpu_gemm(const Dtype* weight, const Dtype* gate_diff, Dtype* data_diff, bool isWX);



		int T_;			//序列长度
		int batch_size_;	//batch大小

		vector<int> kernel_shape_;
		vector<int> kernel_shape_hx_;
		Blob<int> stride_;
		Blob<int> pad_;
		Blob<int> dilation_;
		vector<int> wh_pad_;
		vector<int> wh_stride_;
	

		bool bias_term_;					//是否需要bias

		
		int conv_out_channels_;
		int kernel_height_;
		int kernel_width_;
		int kernel_dim_;    //卷积核的c*h*w
		int kernel_dim_hx_;


		const vector<int>* bottom_shape_;		//bottom的形状
		vector<int> top_shape_;					//top的形状
		int conv_out_spatial_dim_;				//top的h*w
		int bottom_dim_;						//bottom的c*h*w
		int top_dim_;							//top的c*h*w
		vector<int> output_shape_;


		vector<int> col_buffer_shape_;		//列缓存的shape
		Blob<Dtype> col_buffer_;				//列缓存
		vector<int> col_buffer_hx_shape_;
		Blob<Dtype> col_buffer_hx_;

		//rnn参数
		vector<shared_ptr<Blob<Dtype>>> gate_;
		vector<shared_ptr<Blob<Dtype>>> pre_gate_;
		Blob<Dtype> cell_;
		Blob<Dtype> h_to_h_;

	private:
		int num_;
		Blob<Dtype> bias_multiplier_;			//偏移的乘数
		inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff , bool isWX) {
			if (isWX)
			{
				im2col_cpu(data, (*bottom_shape_)[1],
					(*bottom_shape_)[2], (*bottom_shape_)[3],
					kernel_shape_[2], kernel_shape_[3],
					pad_.cpu_data()[0], pad_.cpu_data()[1],
					stride_.cpu_data()[0], stride_.cpu_data()[1],
					dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
			}
			else
			{
				im2col_cpu(data, top_shape_[1],
					top_shape_[2], top_shape_[3],
					kernel_shape_[2], kernel_shape_[3],
					wh_pad_[0], wh_pad_[1],
					wh_stride_[0], wh_stride_[1],
					dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);

			}
			
		}
		inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data,bool isWX) {	
			if (isWX)
			{
				col2im_cpu(col_buff, (*bottom_shape_)[1],
					(*bottom_shape_)[2], (*bottom_shape_)[3],
					kernel_shape_[2], kernel_shape_[3],
					pad_.cpu_data()[0], pad_.cpu_data()[1],
					stride_.cpu_data()[0], stride_.cpu_data()[1],
					dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
			}
			else
			{
				col2im_cpu(col_buff, top_shape_[1],
					top_shape_[2], top_shape_[3],
					kernel_shape_[2], kernel_shape_[3],
					wh_pad_[0], wh_pad_[1],
					wh_stride_[0], wh_stride_[1],
					dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);

			}

		}
	};

}