#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_lstm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/common_math.hpp"
#include <iostream>
using namespace std;
namespace caffe
{

	//ÐÞ¸Äº¯ÊýÉùÃ÷¡£¡£¡£¡£¡£¡£¡£¡£
	template <typename Dtype>
	void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
		const Blob<Dtype>* weights,
		Blob<Dtype>* out, int t, bool isWX) {
		CHECK_EQ(4, out->num_axes());
		// Kernel size, stride, and pad
		int kernel_h, kernel_w;
		if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
			kernel_h = conv_param->kernel_h();
			kernel_w = conv_param->kernel_w();
		}
		else {
			kernel_h = kernel_w = conv_param->kernel_size(0);
		}
		int pad_h, pad_w;
		if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
			pad_h = conv_param->pad_h();
			pad_w = conv_param->pad_w();
		}
		else {
			pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
		}
		int stride_h, stride_w;
		if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
			stride_h = conv_param->stride_h();
			stride_w = conv_param->stride_w();
		}
		else {
			stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
		}
		int dilation_h, dilation_w;
		dilation_h = dilation_w = conv_param->dilation_size() ?
			conv_param->dilation(0) : 1;

		// Groups
		int o_g = out->shape(1);
		int k_g = in->shape(1);
		// Convolution
		vector<int> weight_offset(4);
		vector<int> in_offset(4);
		vector<int> out_offset(4);
		Dtype* out_data = out->mutable_cpu_data();
		for (int o = 0; o < o_g; o++) {
			for (int k = 0; k < k_g; k++) {
				for (int y = 0; y < out->shape(2); y++) {
					for (int x = 0; x < out->shape(3); x++) {
						for (int p = 0; p < kernel_h; p++) {
							for (int q = 0; q < kernel_w; q++) {
								int in_y = y * stride_h - pad_h + p * dilation_h;
								int in_x = x * stride_w - pad_w + q * dilation_w;
								if (true
									&& in_y >= 0 && in_y < in->shape(2)
									&& in_x >= 0 && in_x < in->shape(3)) {

									weight_offset[0] = o;
									weight_offset[1] = k;
									weight_offset[2] = p;
									weight_offset[3] = q;

									if (isWX)
									{
										in_offset[0] = t;
										out_offset[0] = t;
									}
									else
									{
										in_offset[0] = t - 1;
										out_offset[0] = 0;
									}

									in_offset[1] = k;
									in_offset[2] = in_y;
									in_offset[3] = in_x;


									out_offset[1] = o;
									out_offset[2] = y;
									out_offset[3] = x;
									out_data[out->offset(out_offset)] += in->data_at(in_offset) * weights->data_at(weight_offset);

								}
							}
						}

					}
				}
			}
		}


	}

	template void caffe_conv(const Blob<float>* in, ConvolutionParameter* conv_param,
		const Blob<float>* weights,
		Blob<float>* out, int t, bool isWX);
	template void caffe_conv(const Blob<double>* in, ConvolutionParameter* conv_param,
		const Blob<double>* weights,
		Blob<double>* out, int t, bool isWX);

	template <typename Dtype>
	void caffe_bias(const Blob<Dtype>* weights, Blob<Dtype>* out, int t)
	{
		// Bias
		Dtype* out_data = out->mutable_cpu_data();
		vector<int> out_offset(4);
		const Dtype* bias_data = weights->cpu_data();
		for (int o = 0; o < out->shape(1); o++) {
			for (int y = 0; y < out->shape(2); y++) {
				for (int x = 0; x < out->shape(3); x++) {
					out_offset[0] = t;
					out_offset[1] = o;
					out_offset[2] = y;
					out_offset[3] = x;
					out_data[out->offset(out_offset)] += bias_data[o];
				}

			}
		}
	}

	template void caffe_bias(const Blob<float>* weights, Blob<float>* out, int t);
	template void caffe_bias( const Blob<double>* weights, Blob<double>* out, int t);


	template <typename Dtype>
	void caffe_conv_lstm(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
		const vector<shared_ptr<Blob<Dtype> > >& weights,
		Blob<Dtype>* out)
	{

		int top_dim = out->count(1);
		const vector<int>& shape = out->shape();
		ConvolutionParameter convolution_param_2;
		convolution_param_2.add_kernel_size(3);
		convolution_param_2.add_stride(1);
		convolution_param_2.set_num_output(12);
		convolution_param_2.set_pad_h(1);
		convolution_param_2.set_pad_w(1);
		int T_ = in->num();
		vector<shared_ptr<Blob<Dtype> > > gate;
		vector<shared_ptr<Blob<Dtype> > > pre_gate;
		Blob<Dtype> cell;


		vector<int> gate_shape;
		gate_shape.push_back(1);
		gate_shape.push_back(conv_param->num_output());
		gate_shape.push_back(out->shape(2));
		gate_shape.push_back(out->shape(3));
		gate.resize(4);
		pre_gate.resize(4);

		const vector<int>& output_shape = out->shape();
		cell.Reshape(output_shape);

		for (int i = 0; i < 4; i++)
		{
			gate[i].reset(new Blob<Dtype>(output_shape));
			pre_gate[i].reset(new Blob<Dtype>(gate_shape));
		}


		const Dtype* pre_cell_data = NULL;
		const Dtype* pre_h_data = NULL;

		for (int t = 0; t < T_; t++)
		{
			Dtype* h_data = out->mutable_cpu_data() + out->offset(t);
			Dtype* cell_data = cell.mutable_cpu_data() + cell.offset(t);
			if (t > 0)
			{
				pre_cell_data = cell.cpu_data() + cell.offset((t - 1));
				pre_h_data = out->cpu_data() + out->offset((t - 1));
			}
			for (int i = 0; i < 4; i++)
			{


				caffe_conv(in, conv_param, weights[i].get(), gate[i].get(), t, true);
				if (t > 0)
				{
					caffe_conv(out, &convolution_param_2, weights[i + 4].get(), pre_gate[i].get(), t, false);
					Dtype* gate_data = gate[i]->mutable_cpu_data() + gate[i]->offset(t);
					Dtype* pre_gate_data = pre_gate[i]->mutable_cpu_data();
					for (int n = 0; n < top_dim; n++)
					{
						gate_data[n] += pre_gate_data[n];
					}
				}
				if (conv_param->bias_term())
				{
					caffe_bias(weights[8 + i].get(), gate[i].get(), t);
				}
			}


			for (int i = 0; i < 4; i++)
			{
				//gate_data = sigmoid(gate_data)
				Dtype* gate_data = gate[i]->mutable_cpu_data() + gate[i]->offset(t);
				for (int n = 0; n < top_dim; n++)
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


			const Dtype* gt = gate[3]->cpu_data() + gate[3]->offset(t);
			const Dtype* it = gate[0]->cpu_data() + gate[0]->offset(t);
			const Dtype* ft = gate[1]->cpu_data() + gate[1]->offset(t);
			const Dtype* ot = gate[2]->cpu_data() + gate[2]->offset(t);
			for (int n = 0; n < top_dim; n++)
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

	template void caffe_conv_lstm(const Blob<float>* in, ConvolutionParameter* conv_param,
		const vector<shared_ptr<Blob<float> > >& weights,
		Blob<float>* out);
	template void caffe_conv_lstm(const Blob<double>* in, ConvolutionParameter* conv_param,
		const vector<shared_ptr<Blob<double> > >& weights,
		Blob<double>* out);



	template <typename TypeParam>
	class ConvolutionLSTMLayerTest : public MultiDeviceTest < TypeParam > {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		ConvolutionLSTMLayerTest()
			: blob_bottom_(new Blob<Dtype>(2, 3, 8, 8)),
			blob_top_(new Blob<Dtype>()){}
		virtual void SetUp() {
			// fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~ConvolutionLSTMLayerTest() {
			delete blob_bottom_;
			delete blob_top_;
			cout << "ConvolutionLSTMLayerTest" << endl;
		}

		virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
			this->ref_blob_top_.reset(new Blob<Dtype>());
			this->ref_blob_top_->ReshapeLike(*top);
			return this->ref_blob_top_.get();
		}

		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		shared_ptr<Blob<Dtype> > ref_blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ConvolutionLSTMLayerTest, TestDtypesAndDevices);


	TYPED_TEST(ConvolutionLSTMLayerTest, TestConvolutionLSTM)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param_1;
		ConvolutionParameter* convolution_param =
			layer_param_1.mutable_convolution_param();
		convolution_param->add_kernel_size(3);
		convolution_param->add_stride(1);
		convolution_param->set_num_output(12);
		//convolution_param->mutable_bias_filler()->set_type("constant");
		//convolution_param->mutable_bias_filler()->set_value(1.0);
		convolution_param->set_bias_term(false);

		shared_ptr<Layer<Dtype> > layer(
			new ConvolutionLSTMLayer<Dtype>(layer_param_1));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		//layer->Forward_cpu(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		const Dtype* top_data;
		const Dtype* ref_top_data;
		caffe_conv_lstm(this->blob_bottom_, convolution_param, layer->blobs(), this->MakeReferenceTop(this->blob_top_));
		top_data = this->blob_top_->cpu_data();
		ref_top_data = this->ref_blob_top_->cpu_data();
		for (int i = 0; i < this->blob_top_->count(); ++i) {
			EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-3);
		}
	}
	//Test LayerSetUp

	TYPED_TEST(ConvolutionLSTMLayerTest, TestGradient)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param =
			layer_param.mutable_convolution_param();
		//this->blob_bottom_vec_.push_back(this->blob_bottom_);
		//this->blob_top_vec_.push_back(this->blob_top_);
		convolution_param->add_kernel_size(3);
		convolution_param->add_stride(1);
		convolution_param->set_num_output(12);
		convolution_param->mutable_weight_filler()->set_type("gaussian");
		convolution_param->set_bias_term(false);
		ConvolutionLSTMLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}



}