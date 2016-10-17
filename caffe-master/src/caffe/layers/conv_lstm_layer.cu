#include "caffe/layers/conv_lstm_layer.hpp"
#include <vector>

namespace caffe
{
	template <typename Dtype>
	__device__ Dtype sigmoid(const Dtype x) 
	{
		return Dtype(1) / (Dtype(1) + exp(-x));
	}

	template <typename Dtype>
	__device__ Dtype tanh(const Dtype x) 
	{
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}

	template <typename Dtype>
	__global__ void ComputeGateData(Dtype* gate_data,int top_dim,int i)
	{
		CUDA_KERNEL_LOOP(n, top_dim)
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

	template <typename Dtype>
	__global__ void ComputeHData(const Dtype* pre_cell_data, const Dtype* gt, const Dtype* it, const Dtype* ft, const Dtype* ot, Dtype* cell_data, Dtype* h_data, int t, int top_dim)
	{
		CUDA_KERNEL_LOOP(n, top_dim)
		{
			cell_data[n] = gt[n] * it[n];
			if (t > 0)
			{
				cell_data[n] += pre_cell_data[n] * ft[n];
			}
			h_data[n] = tanh(cell_data[n]) * ot[n];
		}
	}

	template <typename Dtype>
	__global__ void ComputeGateDiff(const Dtype* gt, const Dtype* it, const Dtype* ft, const Dtype* ot,
		const Dtype* cell_data, const Dtype* cell_data_t_1,
		Dtype* pre_gt_diff, Dtype* pre_it_diff, Dtype* pre_ft_diff, Dtype* pre_ot_diff,
		Dtype* gt_diff, Dtype* it_diff, Dtype* ft_diff, Dtype* ot_diff,
		Dtype* cell_diff, Dtype* cell_diff_t_1,
		Dtype* h_diff, int t, int top_dim)
	{

		CUDA_KERNEL_LOOP(n, top_dim)
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
	}
	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		for (int t = 0; t < T_; t++)
		{
			const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(t * batch_size_);
			const Dtype* pre_cell_data = NULL;
			const Dtype* pre_h_data = NULL;

			if (t > 0)
			{
				pre_h_data = top[0]->gpu_data() + top[0]->offset((t - 1) * batch_size_);
			}

			for (int b = 0; b < batch_size_; b++)
			{
				for (int i = 0; i < 4; i++)
				{
					const Dtype* weightX = this->blobs_[i]->gpu_data();
					const Dtype* weightH = this->blobs_[i + 4]->gpu_data();
					Dtype* gate_data = gate_[i]->mutable_gpu_data() + gate_[i]->offset(t * batch_size_);
					Dtype* pre_gate_data = pre_gate_[i]->mutable_gpu_data();

					//卷积操作
					//gate_data =  W * X 
					this->forward_gpu_gemm(bottom_data + b * bottom_dim_, weightX, gate_data + b * top_dim_, true);
					if (t > 0)
					{

						//pre_gate_data = W * H(t-1)
						this->forward_gpu_gemm(pre_h_data + b * top_dim_, weightH, pre_gate_data + b * top_dim_, false);
						//gate_data = gate_data + pre_gate_data
						caffe_gpu_add(top_dim_, pre_gate_data + b * top_dim_, gate_data + b * top_dim_, gate_data + b * top_dim_);

					}

					if (bias_term_)
					{
						const Dtype* bias = this->blobs_[8 + i]->gpu_data();
						caffe_gpu_add(top_dim_, bias, gate_data + b * top_dim_, gate_data + b * top_dim_);
					}

				}
				/////////////////////////////////////////////////////////////////////////////////////////////////////////////
				for (int i = 0; i < 4; i++)
				{
					//gate_data = sigmoid(gate_data)
					Dtype* gate_data = gate_[i]->mutable_gpu_data() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
					ComputeGateData<Dtype> << <CAFFE_GET_BLOCKS(top_dim_), CAFFE_CUDA_NUM_THREADS >> >(gate_data,top_dim_,i);
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//计算cell
				//计算h
				const Dtype* gt = gate_[3]->gpu_data() + gate_[3]->offset(b) + gate_[3]->offset(t * batch_size_);
				const Dtype* it = gate_[0]->gpu_data() + gate_[0]->offset(b) + gate_[0]->offset(t * batch_size_);
				const Dtype* ft = gate_[1]->gpu_data() + gate_[1]->offset(b) + gate_[1]->offset(t * batch_size_);
				const Dtype* ot = gate_[2]->gpu_data() + gate_[2]->offset(b) + gate_[2]->offset(t * batch_size_);

				Dtype* cell_data = cell_.mutable_gpu_data() + cell_.offset(t * batch_size_) + cell_.offset(b);
				Dtype* h_data = top[0]->mutable_gpu_data() + top[0]->offset(t * batch_size_) + top[0]->offset(b);
				if (t > 0)
				{
					pre_cell_data = cell_.gpu_data() + cell_.offset((t - 1) * batch_size_) + cell_.offset(b);
				}
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				ComputeHData << <CAFFE_GET_BLOCKS(top_dim_), CAFFE_CUDA_NUM_THREADS >> >(pre_cell_data, gt, it, ft, ot, cell_data, h_data, t, top_dim_);
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
			}
		}
	}

	template <typename Dtype>
	void ConvolutionLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		//梯度清零
		caffe_gpu_set(cell_.count(), Dtype(0.), cell_.mutable_gpu_diff());
		for (int i = 0; i < 4; i++)
		{
			caffe_gpu_set(gate_[i]->count(), Dtype(0.), gate_[i]->mutable_gpu_diff());
			caffe_gpu_set(pre_gate_[i]->count(), Dtype(0.), pre_gate_[i]->mutable_gpu_diff());

		}
		for (int t = T_ - 1; t >= 0; t--)
		{

			const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(t * batch_size_);
			Dtype* h_diff = top[0]->mutable_gpu_diff() + top[0]->offset(t * batch_size_);
			Dtype* cell_diff = cell_.mutable_gpu_diff() + cell_.offset(t * batch_size_);

			const Dtype* cell_data_t_1 = t > 0 ? cell_.gpu_data() + cell_.offset((t - 1) * batch_size_) : NULL;
			const Dtype* cell_data = cell_.gpu_data() + cell_.offset(t * batch_size_);

			Dtype* cell_diff_t_1 = t > 0 ? cell_.mutable_gpu_diff() + cell_.offset((t - 1) * batch_size_) : NULL;

			const Dtype* gt = gate_[3]->gpu_data() + gate_[3]->offset(t * batch_size_);
			const Dtype* it = gate_[0]->gpu_data() + gate_[0]->offset(t * batch_size_);
			const Dtype* ft = gate_[1]->gpu_data() + gate_[1]->offset(t * batch_size_);
			const Dtype* ot = gate_[2]->gpu_data() + gate_[2]->offset(t * batch_size_);

			Dtype* pre_gt_diff = pre_gate_[3]->mutable_gpu_diff();
			Dtype* pre_it_diff = pre_gate_[0]->mutable_gpu_diff();
			Dtype* pre_ft_diff = pre_gate_[1]->mutable_gpu_diff();
			Dtype* pre_ot_diff = pre_gate_[2]->mutable_gpu_diff();

			Dtype* gt_diff = gate_[3]->mutable_gpu_diff() + gate_[3]->offset(t * batch_size_);
			Dtype* it_diff = gate_[0]->mutable_gpu_diff() + gate_[0]->offset(t * batch_size_);
			Dtype* ft_diff = gate_[1]->mutable_gpu_diff() + gate_[1]->offset(t * batch_size_);
			Dtype* ot_diff = gate_[2]->mutable_gpu_diff() + gate_[2]->offset(t * batch_size_);

			for (int b = 0; b < batch_size_; b++)
			{
				//////////////////////////////////////////////////////////////////////////////////////////////////
				//计算gate_diff
				ComputeGateDiff << <CAFFE_GET_BLOCKS(top_dim_), CAFFE_CUDA_NUM_THREADS >> >(gt, it, ft, ot, cell_data, cell_data_t_1,
					pre_gt_diff, pre_it_diff, pre_ft_diff, pre_ot_diff,
					gt_diff, it_diff, ft_diff, ot_diff,
					cell_diff, cell_diff_t_1,
					h_diff, t, top_dim_);
				/////////////////////////////////////////////////////////////////////////////////////////////////////////

				//计算ht_diff
				if (t > 0)
				{
					Dtype* h_diff_t_1 = top[0]->mutable_gpu_diff() + top[0]->offset((t - 1)*batch_size_) + top[0]->offset(b);
					Dtype* col_buffer = col_buffer_hx_.mutable_gpu_diff();
					caffe_gpu_set(col_buffer_hx_.count(), static_cast<Dtype>(0), col_buffer);
					Dtype* h_to_h = h_to_h_.mutable_gpu_data();
					for (int i = 0; i < 4; i++)
					{
						const Dtype* weightH = this->blobs_[i + 4]->gpu_data();
						const Dtype* gate_diff = gate_[i]->gpu_diff() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
						caffe_gpu_gemm(CblasTrans, CblasNoTrans, kernel_dim_hx_, conv_out_spatial_dim_, conv_out_channels_, Dtype(1.), weightH, gate_diff, Dtype(1.), col_buffer);
					}
					conv_col2im_gpu(col_buffer, h_to_h, false);
					caffe_gpu_add(top_dim_, h_to_h, h_diff_t_1, h_diff_t_1);
					cell_data_t_1 += cell_.offset(1);
					cell_diff_t_1 += cell_.offset(1);
				}
				//计算权值diff
				for (int i = 0; i < 4; i++)
				{
					const Dtype* gate_diff = gate_[i]->gpu_diff() + gate_[i]->offset(b) + gate_[i]->offset(t * batch_size_);
					Dtype* Wx_diff = this->blobs_[i]->mutable_gpu_diff();
					weight_gpu_gemm(bottom_data, gate_diff, Wx_diff, true);

					if (t > 0)
					{
						const Dtype* h_data_1 = top[0]->gpu_data() + top[0]->offset((t - 1) * batch_size_) + top[0]->offset(b);
						Dtype* Wh_diff = this->blobs_[i + 4]->mutable_gpu_diff();
						weight_gpu_gemm(h_data_1, gate_diff, Wh_diff, false);
					}
					if (bias_term_)
					{
						Dtype* bais_diff = this->blobs_[i + 8]->mutable_gpu_diff();
						caffe_gpu_add(top_dim_, gate_diff, bais_diff, bais_diff);
					}
				}



				//计算x_diff
				Dtype* data_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(t * batch_size_) + bottom[0]->offset(b);
				Dtype* col_buffer = col_buffer_.mutable_gpu_diff();
				caffe_gpu_set(col_buffer_.count(), static_cast<Dtype>(0), col_buffer);
				for (int i = 0; i < 4; i++)
				{
					const Dtype* weight = this->blobs_[i]->gpu_data();
					const Dtype* gate_diff = gate_[i]->gpu_diff() + gate_[i]->offset(t * batch_size_) + gate_[i]->offset(b);
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_, Dtype(1.), weight, gate_diff, Dtype(1.), col_buffer);
				}
				conv_col2im_gpu(col_buffer, data_diff, true);



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

	INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLSTMLayer);
}