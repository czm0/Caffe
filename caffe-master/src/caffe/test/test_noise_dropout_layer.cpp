#include <vector>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/noise_dropout_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include <iostream>
using namespace std;
namespace caffe
{
	template <typename TypeParam>
	class NoiseDropoutLayerTest : public MultiDeviceTest < TypeParam > {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		NoiseDropoutLayerTest()
			: blob_bottom_(new Blob<Dtype>(1, 1, 6, 6)),
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

		virtual ~NoiseDropoutLayerTest() {
			delete blob_bottom_;
			delete blob_top_;
			cout << "NoiseDropoutLayerTest" << endl;
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


	TYPED_TEST_CASE(NoiseDropoutLayerTest, TestDtypesAndDevices);

	TYPED_TEST(NoiseDropoutLayerTest, TestNoiseDropout)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		NoiseDropoutParameter noise_dropout_param =  layer_param.noise_dropout_param();
		noise_dropout_param.set_dropout_ratio(0.5);
		shared_ptr<Layer<Dtype> > layer(new NoiseDropoutLayer<Dtype>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		for (int i = 0; i < this->blob_bottom_vec_.size(); i++)
		{
			const int count = this->blob_bottom_vec_[i]->count();
			const Dtype* bottom_data = this->blob_bottom_vec_[i]->cpu_data();
			const Dtype* top_data = this->blob_top_vec_[i]->cpu_data();
			for (int n = 0; n < count; n++)
			{
				CHECK((bottom_data[n] == top_data[n] || top_data[n] == 0));		
			}
		}

	}

}