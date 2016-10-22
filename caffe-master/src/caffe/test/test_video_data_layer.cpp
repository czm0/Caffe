#include <vector>
#include <cstring>
#include <sstream>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
namespace caffe
{
	template <typename TypeParam>
	class VideoDataLayerTest : public MultiDeviceTest < TypeParam > {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		VideoDataLayerTest() 
			:filename_("test.txt"),
			blob_top_data_(new Blob<Dtype>()) 
			//blob_top_label_(new Blob<Dtype>()),
			{}
		virtual void SetUp() {
			blob_top_vec_.push_back(blob_top_data_);
			//blob_top_vec_.push_back(blob_top_label_);
		}
		virtual ~VideoDataLayerTest() {
			delete blob_top_data_;
			//delete blob_top_label_;
			cout << "VideoDataLayerTest" << endl;
		}
		void outputImage(string image_prefix)
		{
			cv::Mat image(this->blob_top_data_->width(), this->blob_top_data_->height(), CV_8UC3, cv::Scalar(0, 0, 0));
			const Dtype* top_data = this->blob_top_data_->cpu_data();
			vector<int> shape(4);
			string imageName;
			stringstream ss;
			for (int n = 0; n < this->blob_top_data_->num(); n++)
			{
				for (int c = 0; c < this->blob_top_data_->channels(); c++)
				{
					for (int h = 0; h < this->blob_top_data_->height(); h++)
					{
						for (int w = 0; w < this->blob_top_data_->width(); w++)
						{
							shape[0] = n;
							shape[1] = c;
							shape[2] = h;
							shape[3] = w;

							image.at<cv::Vec3b>(h, w)[c] = static_cast<unsigned char>(top_data[this->blob_top_data_->offset(shape)]);
						}
					}
				}
				ss.str("");
				ss.clear();
				ss << image_prefix << n << ".jpg";
				imageName = ss.str();
				cv::imwrite(imageName, image);
			}
		}

		string filename_;
		Blob<Dtype>* const blob_top_data_;
		//Blob<Dtype>* const blob_top_label_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};


	TYPED_TEST_CASE(VideoDataLayerTest, TestDtypesAndDevices);

	TYPED_TEST(VideoDataLayerTest, Test) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		param.set_phase(TRAIN);
		VideoDataParameter* video_data_param = param.mutable_video_data_param();
		video_data_param->set_shuffle(true);
		video_data_param->set_step(4);
		video_data_param->set_source(this->filename_.c_str());
		video_data_param->set_new_height(220);
		video_data_param->set_new_width(220);
		TransformationParameter* transform_param = param.mutable_transform_param();
		transform_param->set_crop_size(200);
		transform_param->set_mirror(false);

		shared_ptr<Layer<Dtype> > layer(new VideoDataLayer<Dtype>(param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		this->outputImage("image_test_1-");
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		this->outputImage("image_test_2-");
	}
}