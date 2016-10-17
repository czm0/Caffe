#include "caffe/layers/conv_lstm_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <vector>
#include <fstream>
using namespace std;
namespace caffe
{
	template <typename Dtype>
	inline Dtype sigmoid(Dtype x) {
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		int channels = bottom[0]->channels();
		ConvolutionLstmParameter conv_lstm_param = this->layer_param_.conv_lstm_param();
		batch_size_ = conv_lstm_param.batch_size();
		conv_out_channels_ = conv_lstm_param.num_output();
		kernel_height_ = conv_lstm_param.kernel_size();
		kernel_width_ = conv_lstm_param.kernel_size();
		bias_term_ = conv_lstm_param.bias_term();

		CHECK_EQ(conv_lstm_param.kernel_size() % 2, 1) << "kernel_size should be an odd number!" << endl;

		wh_pad_.clear();
		wh_pad_.push_back(kernel_height_ / 2);
		wh_pad_.push_back(kernel_height_ / 2);
		wh_stride_.clear();
		wh_stride_.push_back(1);
		wh_stride_.push_back(1);
		vector<int> spatial_dim_blob_shape(1, 2);
		pad_.Reshape(spatial_dim_blob_shape);
		int* pad_data = pad_.mutable_cpu_data();
		pad_data[0] = conv_lstm_param.pad();
		pad_data[1] = conv_lstm_param.pad();

		stride_.Reshape(spatial_dim_blob_shape);
		int* stride_data = stride_.mutable_cpu_data();
		stride_data[0] = conv_lstm_param.stride();
		stride_data[1] = conv_lstm_param.stride();

		dilation_.Reshape(spatial_dim_blob_shape);
		int* dilation_data = dilation_.mutable_cpu_data();
		dilation_data[0] = conv_lstm_param.dilation();
		dilation_data[1] = conv_lstm_param.dilation();

		kernel_shape_.clear();
		kernel_shape_.push_back(conv_out_channels_);
		kernel_shape_.push_back(channels);
		kernel_shape_.push_back(kernel_height_);
		kernel_shape_.push_back(kernel_width_);


		kernel_shape_hx_.clear();
		kernel_shape_hx_.push_back(conv_out_channels_);
		kernel_shape_hx_.push_back(conv_out_channels_);
		kernel_shape_hx_.push_back(kernel_height_);
		kernel_shape_hx_.push_back(kernel_width_);
	
		//初始化权值
		if (bias_term_)
		{
			this->blobs_.resize(12);
		}
		else
		{
			this->blobs_.resize(8);
		}

		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
			conv_lstm_param.weight_filler()));
		for (int i = 0; i < 4; i++)
		{
			this->blobs_[i].reset(new Blob<Dtype>(kernel_shape_));
			this->blobs_[i + 4].reset(new Blob<Dtype>(kernel_shape_hx_));
			weight_filler->Fill(this->blobs_[i].get());
			weight_filler->Fill(this->blobs_[i + 4].get());
		}

		bottom_shape_ = &bottom[0]->shape();
		compute_output_shape();

		//初始化偏移
		if (bias_term_)
		{
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
				conv_lstm_param.bias_filler()));

			vector<int> bias_shape;
			bias_shape.push_back(conv_out_channels_);
			bias_shape.push_back(output_shape_[0]);
			bias_shape.push_back(output_shape_[1]);
			for (int i = 0; i < 4; i++)
			{
				this->blobs_[8 + i].reset(new Blob<Dtype>(bias_shape));
				bias_filler->Fill(this->blobs_[8 + i].get());
			}
		}
		kernel_dim_ = this->blobs_[0]->count(1);
		kernel_dim_hx_ = this->blobs_[4]->count(1);

		this->param_propagate_down_.resize(this->blobs_.size(), true);	//计算diff
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		num_ = bottom[0]->num();

		CHECK_EQ(num_ % batch_size_, 0) << "Number of num_ should be multiples of batch_size" << endl;

		T_ = num_ / batch_size_;
		top_shape_.clear();
		top_shape_.push_back(num_);
		top_shape_.push_back(conv_out_channels_);
		top_shape_.push_back(output_shape_[0]);
		top_shape_.push_back(output_shape_[1]);
		top[0]->Reshape(top_shape_);
		top_dim_ = top[0]->count(1);
		bottom_dim_ = bottom[0]->count(1);
		conv_out_spatial_dim_ = top[0]->count(2);

		col_buffer_shape_.clear();
		col_buffer_shape_.push_back(kernel_dim_);
		col_buffer_shape_.push_back(output_shape_[0]);
		col_buffer_shape_.push_back(output_shape_[1]);
		col_buffer_.Reshape(col_buffer_shape_);

		col_buffer_hx_shape_.clear();
		col_buffer_hx_shape_.push_back(kernel_dim_hx_);
		col_buffer_hx_shape_.push_back(output_shape_[0]);
		col_buffer_hx_shape_.push_back(output_shape_[1]);
		col_buffer_hx_.Reshape(col_buffer_hx_shape_);

		vector<int> gate_shape;
		gate_shape.push_back(batch_size_);
		gate_shape.push_back(conv_out_channels_);
		gate_shape.push_back(output_shape_[0]);
		gate_shape.push_back(output_shape_[1]);
		gate_.resize(4);
		pre_gate_.resize(4);
		for (int i = 0; i < 4; i++)
		{
			gate_[i].reset(new Blob<Dtype>(top_shape_));
			pre_gate_[i].reset(new Blob<Dtype>(gate_shape));
		}
		cell_.Reshape(top_shape_);
		h_to_h_.Reshape(1, top_shape_[1], top_shape_[2], top_shape_[3]);
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		for (int t = 0; t < T_; t++)
		{
			const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(t * batch_size_);
			const Dtype* pre_cell_data = NULL;
			const Dtype* pre_h_data = NULL;

			if (t > 0)
			{
				pre_h_data = top[0]->cpu_data() + top[0]->offset((t - 1) * batch_size_);
			}

			for (int b = 0; b < batch_size_; b++)
			{
				for (int i = 0; i < 4; i++)
				{
					const Dtype* weightX = this->blobs_[i]->cpu_data();
					const Dtype* weightH = this->blobs_[i + 4]->cpu_data();
					Dtype* gate_data = gate_[i]->mutable_cpu_data() + gate_[i]->offset(t * batch_size_);
					Dtype* pre_gate_data = pre_gate_[i]->mutable_cpu_data();

					//卷积操作
					//gate_data =  W * X 
					this->forward_cpu_gemm(bottom_data + b * bottom_dim_, weightX, gate_data + b * top_dim_, true);
					if (t > 0)
					{
						//pre_gate_data = W * H(t-1)
						this->forward_cpu_gemm(pre_h_data + b * top_dim_, weightH, pre_gate_data + b * top_dim_, false);
						//gate_data = gate_data + pre_gate_data
						caffe_add(top_dim_, pre_gate_data + b * top_dim_, gate_data + b * top_dim_, gate_data + b * top_dim_);
					}

					if (bias_term_)
					{
						const Dtype* bias = this->blobs_[8 + i]->cpu_data();
						caffe_add(top_dim_, bias, gate_data + b * top_dim_, gate_data + b * top_dim_);
					}
				}

				for (int i = 0; i < 4; i++)
				{
					//gate_data = sigmoid(gate_data)
					Dtype* gate_data = gate_[i]->mutable_cpu_data() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
					for (int n = 0; n < top_dim_; n++)
					{
						if (i == 3)
						{
							gate_data[n] = tanh(gate_data[n]);
						}
						else
						{
							gate_data[n] = sigmoid(gate_data[n]);
						}
					}
				}
				//计算cell
				//计算h
				const Dtype* gt = gate_[3]->cpu_data() + gate_[3]->offset(b) + gate_[3]->offset(t * batch_size_);
				const Dtype* it = gate_[0]->cpu_data() + gate_[0]->offset(b) + gate_[0]->offset(t * batch_size_);
				const Dtype* ft = gate_[1]->cpu_data() + gate_[1]->offset(b) + gate_[1]->offset(t * batch_size_);
				const Dtype* ot = gate_[2]->cpu_data() + gate_[2]->offset(b) + gate_[2]->offset(t * batch_size_);

				Dtype* cell_data = cell_.mutable_cpu_data() + cell_.offset(t * batch_size_) + cell_.offset(b);
				Dtype* h_data = top[0]->mutable_cpu_data() + top[0]->offset(t * batch_size_) + top[0]->offset(b);
				if (t > 0)
				{
					pre_cell_data = cell_.cpu_data() + cell_.offset((t - 1) * batch_size_) + cell_.offset(b);
				}
				for (int n = 0; n < top_dim_; n++)
				{
					cell_data[n] = gt[n] * it[n];
					if (t > 0)
					{
						cell_data[n] += pre_cell_data[n] * ft[n];
					}
					h_data[n] = tanh(cell_data[n]) * ot[n];
				}

			}
		}
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		//梯度清零
		caffe_set(cell_.count(), Dtype(0.), cell_.mutable_cpu_diff());
		for (int i = 0; i < 4; i++)
		{
			caffe_set(gate_[i]->count(), Dtype(0.), gate_[i]->mutable_cpu_diff());
			caffe_set(pre_gate_[i]->count(), Dtype(0.), pre_gate_[i]->mutable_cpu_diff());
		}


		for (int t = T_ - 1; t >= 0; t--)
		{
			const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(t * batch_size_);
			Dtype* h_diff = top[0]->mutable_cpu_diff() + top[0]->offset(t * batch_size_);
			Dtype* cell_diff = cell_.mutable_cpu_diff() + cell_.offset(t * batch_size_);
			const Dtype* cell_data_t_1 = t > 0 ? cell_.cpu_data() + cell_.offset((t - 1) * batch_size_) : NULL;
			const Dtype* cell_data = cell_.cpu_data() + cell_.offset(t * batch_size_);
			Dtype* cell_diff_t_1 = t > 0 ? cell_.mutable_cpu_diff() + cell_.offset((t - 1) * batch_size_) : NULL;

			const Dtype* gt = gate_[3]->cpu_data() + gate_[3]->offset(t * batch_size_);
			const Dtype* it = gate_[0]->cpu_data() + gate_[0]->offset(t * batch_size_);
			const Dtype* ft = gate_[1]->cpu_data() + gate_[1]->offset(t * batch_size_);
			const Dtype* ot = gate_[2]->cpu_data() + gate_[2]->offset(t * batch_size_);

			Dtype* pre_gt_diff = pre_gate_[3]->mutable_cpu_diff();
			Dtype* pre_it_diff = pre_gate_[0]->mutable_cpu_diff();
			Dtype* pre_ft_diff = pre_gate_[1]->mutable_cpu_diff();
			Dtype* pre_ot_diff = pre_gate_[2]->mutable_cpu_diff();

			Dtype* gt_diff = gate_[3]->mutable_cpu_diff() + gate_[3]->offset(t * batch_size_);
			Dtype* it_diff = gate_[0]->mutable_cpu_diff() + gate_[0]->offset(t * batch_size_);
			Dtype* ft_diff = gate_[1]->mutable_cpu_diff() + gate_[1]->offset(t * batch_size_);
			Dtype* ot_diff = gate_[2]->mutable_cpu_diff() + gate_[2]->offset(t * batch_size_);
			for (int b = 0; b < batch_size_; b++)
			{
				//计算gate_diff
				for (int n = 0; n < top_dim_; n++)
				{
					const Dtype tanh_c = tanh(cell_data[n]);
					cell_diff[n] += h_diff[n] * (Dtype(1.) - tanh_c * tanh_c) * ot[n];
					if (t > 0)
					{
						cell_diff_t_1[n] = cell_diff[n] * ft[n];
					}

					pre_gt_diff[n] = cell_diff[n] * it[n];
					pre_it_diff[n] = cell_diff[n] * gt[n];
					//pre_ft_diff[n] = t > 0 ? cell_diff[n] * 
					if (t > 0)
					{
						pre_ft_diff[n] = cell_diff[n] * cell_data_t_1[n];
					}
					else
					{
						pre_ft_diff[n] = 0;
					}
					pre_ot_diff[n] = tanh_c * h_diff[n];


					gt_diff[n] = pre_gt_diff[n] * (Dtype(1.0) - gt[n] * gt[n]);
					it_diff[n] = pre_it_diff[n] * it[n] * (Dtype(1.0) - it[n]);
					ft_diff[n] = pre_ft_diff[n] * ft[n] * (Dtype(1.0) - ft[n]);
					ot_diff[n] = pre_ot_diff[n] * ot[n] * (Dtype(1.0) - ot[n]);
				}

				//计算ht_diff
				if (t > 0)
				{
					Dtype* h_diff_t_1 = top[0]->mutable_cpu_diff() + top[0]->offset((t - 1)*batch_size_) + top[0]->offset(b);
					Dtype* col_buffer = col_buffer_hx_.mutable_cpu_diff();
					caffe_set(col_buffer_hx_.count(), static_cast<Dtype>(0), col_buffer);
					Dtype* h_to_h = h_to_h_.mutable_cpu_data();
					for (int i = 0; i < 4; i++)
					{
						const Dtype* weightH = this->blobs_[i + 4]->cpu_data();
						const Dtype* gate_diff = gate_[i]->cpu_diff() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
						caffe_cpu_gemm(CblasTrans, CblasNoTrans, kernel_dim_hx_, conv_out_spatial_dim_, conv_out_channels_, Dtype(1.), weightH, gate_diff, Dtype(1.), col_buffer);
					}
					conv_col2im_cpu(col_buffer, h_to_h, false);
					caffe_add(top_dim_, h_to_h, h_diff_t_1, h_diff_t_1);
					cell_data_t_1 += cell_.offset(1);
					cell_diff_t_1 += cell_.offset(1);
				}
				//计算权值diff
				for (int i = 0; i < 4; i++)
				{
					const Dtype* gate_diff = gate_[i]->cpu_diff() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
					Dtype* Wx_diff = this->blobs_[i]->mutable_cpu_diff();
					weight_cpu_gemm(bottom_data, gate_diff, Wx_diff, true);

					if (t > 0)
					{
						const Dtype* h_data_1 = top[0]->cpu_data() + top[0]->offset((t - 1) * batch_size_) + top[0]->offset(b);
						Dtype* Wh_diff = this->blobs_[i + 4]->mutable_cpu_diff();
						weight_cpu_gemm(h_data_1, gate_diff, Wh_diff, false);
					}
					if (bias_term_)
					{
						Dtype* bais_diff = this->blobs_[i + 8]->mutable_cpu_diff();
						caffe_add(top_dim_, gate_diff, bais_diff, bais_diff);
					}
				}

				

				//计算x_diff
				Dtype* data_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(t * batch_size_) + bottom[0]->offset(b);
				Dtype* col_buffer = col_buffer_.mutable_cpu_diff();
				caffe_set(col_buffer_.count(), static_cast<Dtype>(0), col_buffer);
				for (int i = 0; i < 4; i++)
				{
					const Dtype* weight = this->blobs_[i]->cpu_data();
					const Dtype* gate_diff = gate_[i]->cpu_diff() + gate_[i]->offset(t * batch_size_) + gate_[i]->offset(b);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_, Dtype(1.), weight, gate_diff, Dtype(1.), col_buffer);
				}
				conv_col2im_cpu(col_buffer, data_diff, true);

				gt += gate_[3]->offset(1);
				it += gate_[0]->offset(1);
				ft += gate_[1]->offset(1);
				ot += gate_[2]->offset(1);

				h_diff += top[0]->offset(1);
				cell_diff += cell_.offset(1);
				cell_data += cell_.offset(1);

				pre_gt_diff += pre_gate_[3]->offset(1);
				pre_it_diff += pre_gate_[0]->offset(1);
				pre_ft_diff += pre_gate_[1]->offset(1);
				pre_ot_diff += pre_gate_[2]->offset(1);

				gt_diff += gate_[3]->offset(1);
				it_diff += gate_[0]->offset(1);
				ft_diff += gate_[1]->offset(1);
				ot_diff += gate_[2]->offset(1);

				bottom_data += bottom[0]->offset(1);
			}
		}
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::compute_output_shape() {
		const int* stride_data = this->stride_.cpu_data();
		const int* pad_data = this->pad_.cpu_data();
		const int* dilation_data = this->dilation_.cpu_data();
		this->output_shape_.clear();
		for (int i = 0; i < 2; ++i) {
			// i + 1 to skip channel axis
			const int input_dim = this->input_shape(i + 1);
			const int kernel_extent = dilation_data[i] * (kernel_shape_[i + 2] - 1) + 1;
			const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
				/ stride_data[i] + 1;//计算输出图像的大小
			this->output_shape_.push_back(output_dim);
		}
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool isWX)
	{
		const Dtype* col_buff = input;
		if (isWX)
		{
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data(), isWX);
			col_buff = col_buffer_.cpu_data();
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
				conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights, col_buff,
				(Dtype)0., output);
		}
		else
		{
			conv_im2col_cpu(input, col_buffer_hx_.mutable_cpu_data(), isWX);
			col_buff = col_buffer_hx_.cpu_data();
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
				conv_out_spatial_dim_, kernel_dim_hx_,
				(Dtype)1., weights, col_buff,
				(Dtype)0., output);
		}
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias)
	{
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
			conv_out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
			(Dtype)1., output);
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::weight_cpu_gemm(const Dtype* data, const Dtype* gate_diff, Dtype* weight_diff, bool isWX)
	{
		if (isWX)
		{
			conv_im2col_cpu(data, col_buffer_.mutable_cpu_data(), isWX);
			const Dtype* col_buff = col_buffer_.cpu_data();
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_, kernel_dim_, conv_out_spatial_dim_, Dtype(1.0), gate_diff, col_buff, Dtype(1.), weight_diff);
		}
		else
		{
			conv_im2col_cpu(data, col_buffer_hx_.mutable_cpu_data(), isWX);
			const Dtype* col_buff = col_buffer_hx_.cpu_data();
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_, kernel_dim_hx_, conv_out_spatial_dim_, Dtype(1.0), gate_diff, col_buff, Dtype(1.), weight_diff);
		}

	}



#ifndef CPU_ONLY
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::forward_gpu_gemm(const Dtype* input, const Dtype* weights,Dtype* output, bool isWX)
	{
		const Dtype* col_buff = input;
		if (isWX)
		{
			conv_im2col_gpu(input, col_buffer_.mutable_gpu_data(), isWX);
			col_buff = col_buffer_.gpu_data();
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
				conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights, col_buff,
				(Dtype)0., output);
		}
		else
		{
			conv_im2col_gpu(input, col_buffer_hx_.mutable_gpu_data(), isWX);
			col_buff = col_buffer_hx_.gpu_data();
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
				conv_out_spatial_dim_, kernel_dim_hx_,
				(Dtype)1., weights, col_buff,
				(Dtype)0., output);
		}
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
			conv_out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
			(Dtype)1., output);
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::weight_gpu_gemm(const Dtype* data, const Dtype* gate_diff, Dtype* weight_diff, bool isWX)
	{
		if (isWX)
		{
			conv_im2col_gpu(data, col_buffer_.mutable_gpu_data(), isWX);
			const Dtype* col_buff = col_buffer_.gpu_data();
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_, kernel_dim_, conv_out_spatial_dim_, Dtype(1.0), gate_diff, col_buff, Dtype(1.), weight_diff);
		}
		else
		{
			conv_im2col_gpu(data, col_buffer_hx_.mutable_gpu_data(), isWX);
			const Dtype* col_buff = col_buffer_hx_.gpu_data();
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_, kernel_dim_hx_, conv_out_spatial_dim_, Dtype(1.0), gate_diff, col_buff, Dtype(1.), weight_diff);
		}
	}
#endif


#ifdef CPU_ONLY
	STUB_GPU(ConvolutionLSTMLayer);
#endif
	INSTANTIATE_CLASS(ConvolutionLSTMLayer);
}