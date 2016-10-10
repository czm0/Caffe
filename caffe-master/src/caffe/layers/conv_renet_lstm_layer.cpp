
#include "caffe/layers/conv_renet_lstm_layer.hpp"
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_math.hpp"

namespace caffe
{
	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{

		height_ = 6;
		width_ = 6;
		channels_ = 4;


		LOG(INFO) << "ReNetLSTMLayer " << this->layer_param_.name() << endl;
		CHECK_EQ(bottom[0]->num_axes(), 4);
		/*
		hight_ = this->layer_param_
		width_ = this->layer_param_
		channels_ = this->layer_param_
		*/

		CHECK_EQ(channels_ % 2, 0) << "Channels must be able to be divided exactly by 2" << endl;
		CHECK_EQ(bottom[0]->shape(3) % width_, 0) << "bottom width : " << bottom[0]->shape(3) << endl;
		CHECK_EQ(bottom[0]->shape(2) % height_, 0) << "bottom hight : " << bottom[0]->shape(2) << endl;

		//set renet parameter
		//wx_wh_[0] 表示的是W * X
		//wx_wh_[1] 表示的是W * H(t-1)
		wx_wh_.resize(2);
		LayerParameter layer_param;
		layer_param.set_name("ReNetLayer");
		layer_param.set_type("ReNetLSTM");
		ReNetLSTMParameter renet_param = layer_param.renet_lstm_param();
		renet_param.set_direction(ReNetLSTMParameter_Direction_X_DIR);
		renet_param.set_patch_width(bottom[0]->shape(3) / width_);
		renet_param.set_patch_height(bottom[0]->shape(2) / height_);
		renet_param.set_num_output(channels / 2);
		wx_wh_[0].reset(new ReNetLSTMLayer<Dtype>(layer_param));
		renet_param.set_patch_width(1);
		renet_param.set_patch_height(1);
		renet_param.set_num_output(channels / 2);
		wx_wh_[1].reset(new ReNetLSTMLayer<Dtype>(layer_param));

		

	}


	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		T_ = bottom[0]->num() / batch_size;
		CHECK_EQ(bottom[0]->num() % batch_size, 0) << "Input size "
			"should be multiple of batch size";
		// 把bottom的数据复制四份统一放入一个renet训练
		// bottom_[0]包含 [batch c 4h w]
		data_.resize(1);
		data_[0].reset(new Blob<Dtype>);
		vector<int> data_shape;
		data_shape.resize(4);
		data_shape[0] = batch_size;
		data_shape[1] = bottom[0]->shape(1);
		data_shape[2] = bottom[0]->shape(2) * 4;
		data_shape[3] = bottom[0]->shape(3);
		data_[0]->Reshape(data_shape);
		gate_.resize(1);
		gate_[0].reset(new Blob<Dtype>);
		wx_wh_[0].SetUp(data_, gate_);
		wx_wh_[0].Reshape(data_, gate_);


		vector<int>top_shape;
		top_shape.resize(4);
		top_shape[0] = bottom[0]->num();
		top_shape[1] = channels_;
		top_shape[2] = height_;
		top_shape[3] = width_;
		top[0]->Reshape(top_shape);

		// 把h的数据复制四份统一放入一个renet训练
		// h_[0]包含 [c' 4h' w']
		h_.resize(1);
		h_[0].reset(new Blob<Dtype>);
		vector<int>h_shape;
		h_shape.resize(4);
		h_shape[0] = batch_size_;
		h_shape[1] = channels_;
		h_shape[2] = height_*4;
		h_shape[3] = width_;
		h_[0]->Reshape(h_shape);
		vector<int>cell_shape;
		cell_shape.resize(4);
		cell_shape[0] = bottom[0]->num();
		cell_shape[1] = channels_;
		cell_shape[2] = height_;
		cell_shape[3] = width_;
		cell_.Reshape(cell_shape);
	//	h_t_.Reshape(cell_shape);

		gate_wh_.resize(1);
		gate_wh_[0].reset(new Blob<Dtype>);
		wx_wh_[1].SetUp(h_, gate_wh_);
		wx_wh_[1].Reshape(h_, gate_wh_);

		//把renet里的参数传到这里来
		this->blobs_.resize(3);
		this->blobs_[0] = wx_wh_[0].blobs();
		this->blobs_[1] = wx_wh_[1].blobs();

		this->param_propagate_down_.resize(this->blobs_.size(), true);

	}
	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::CopyData(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, int step_id)
	{
		Blob<Dtype>* h = h_[0]->mutable_cpu_data();
		Blob<Dtype>* data = data_[0]->mutable_cpu_data();
		Blob<Dtype>* h_data = NULL;
		Blob<Dtype>* x_data = bottom[0]->cpu_data() + bottom[0]->offset(step_id*batch_size_);

		//copy h(t-1)
		if (step_id == 0)
		{
			caffe_set<Dtype>(h_[0]->count(), Dtype(0), h);
		}
		else
		{
			h_data = = top[0]->cpu_data() + top[0]->offset((step_id - 1)*batch_size_);
			for (int i = 0; i < 4; i++)
			{
				///可能有错
				caffe_copy(batch_size_*channels_*height_*width_, h_data, h + (i*batch_size_*channels_*height_*width_));

				/*
				for (int h = 0; h < height_; h++)
				{
					for (int n = 0; n < batch_size_; n++)
					{
						for (int c = 0; c < channels_; c++)
						{
							for (int w = 0; w < width_; w++)
							{
								h[( ( n * channels_ + c )  * height_ + h + i * height_) *width_ + w] = h_data[((n*channels_ + c)*height_ + h) * width_ + w];
							}
						}
					}
				}
				*/

			}
		}

		//copy data
		for (int i = 0; i < 4; i++)
		{
			//可能出错
			caffe_copy(batch_size_*channels_*height_*width_, x_data, data + (i*batch_size_*channels_*height_*width_));
		}


	}

	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::ComputeGateData()
	{
		Blob<Dtype>* gate = gate_[0]->mutable_cpu_data();
		Blob<Dtype>* gate_wh = gate_wh_[0]->cpu_data();

		for (int n = 0; n < batch_size_; n++)
		{
			for (int c = 0; c < channels_; c++)
			{
				for (int h = 0; h < 3 * height_; h++)
				{
					for (int w = 0; w < width_; w++)
					{
						gate[gate_[0]->offset(n, c, h, w)] = sigmoid<Dtype>(gate[gate_[0]->offset(n, c, h, w)] + gate_wh[gate_wh_[0]->offset(n, c, h, w)]);
					}

				}

				for (int h = 3 * height_; h < 4 * height_; h++)
				{
					for (int w = 0; w < width_; w++)
					{
						gate[gate_[0]->offset(n, c, h, w)] = tanh<Dtype>(gate[gate_[0]->offset(n, c, h, w)] + gate_wh[gate_wh_[0]->offset(n, c, h, w)]);
					}
				}
			}
		}
	}


	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::ComputeCellData(const vector<Blob<Dtype>*>& top,int step_id)
	{
		Blob<Dtype>* cell_data = cell_.mutable_cpu_data() + cell_.offset(step_id * batch_size_);
		Blob<Dtype>* pre_cell_data = NULL;
		Blob<Dtype>* gate_data = gate_[0]->cpu_data();
		Blob<Dtype>* h_data = top[0]->mutable_cpu_data() + top[0]->offset(step_id * batch_size_);
		if (step_id != 0)
		{
			pre_cell_data = cell_.cpu_data() + cell_.offset((step_id - 1) * batch_size_);
		}
		for (int n = 0; n < batch_size_; n++)
		{
			for (int c = 0; c < channels_; c++)
			{
				for (int h = 0; h < height_; h++)
				{
					for (int w = 0; w < width_; w++)
					{
						cell_data[cell_.offset(n, c, h, w)] = gate_data[gate_[0]->offset(n, c, h + 3 * height_, w)] * gate_data[gate_[0]->offset(n, c, h + height_, w)];
						if (step_id != 0)
						{
							cell_data[cell_.offset(n, c, h, w)] += pre_cell_data[cell_.offset(n, c, h, w)] * gate_data[gate_[0]->offset(n, c, h, w)];
						}
						h_data[top[0]->offset(n, c, h, w)] = tanh(cell_data[cell_.offset(n, c, h, w)])*gate_data[gate_[0]->offset(n, c, h + 2 * height_, w)];
					}
				}
			}
		}

	}


	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		for (int t = 0; t < T_; t++)
		{
			CopyData(bottom, top, t);
			wx_wh_[0].Forward(data_, gate_);
			wx_wh_[1].Forward(h_, gate_wh_);
			ComputeGateData();
			ComputeCellData(top,t);
		}
	}

	template<typename Dtype>
	void ConvolutionReNetLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		for (int t = 0; t < T_; t++)
		{

		}
	}
}
