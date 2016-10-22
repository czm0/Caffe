#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
namespace caffe{
	template <typename Dtype>
	VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
		this->StopInternalThread();
	}

	template <typename Dtype>
	void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		new_height_ = this->layer_param_.video_data_param().new_height();
		new_width_ = this->layer_param_.video_data_param().new_width();
		step_ = this->layer_param_.video_data_param().step();
		const string& source = this->layer_param_.video_data_param().source();
		crop_size_ = this->layer_param_.transform_param().crop_size();
		lines_id_ = 0;
		//读取视屏位置和label信息
		LOG(INFO) << "Opening file: " << source;
		infile.open(source.c_str());
		string filename;
		int label;
		if (this->output_labels_)
		{
			while (infile >> filename>> label){
				video item;
				item.filename = filename;
				item.label = label;
				lines_.push_back(item);
			}
		}
		else
		{
			while (infile >> filename){
				video item;
				item.filename = filename;
				lines_.push_back(item);
			}
		}
		
		//打乱顺序
		if (this->layer_param_.video_data_param().shuffle()){
			const unsigned int prefectch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
			ShuffleVideos();
		}

		LOG(INFO) << "A total of " << lines_.size() << " videos.";
		
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = 3;
		shape[2] = new_height_;
		shape[3] = new_width_;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(shape);
		}
		if (this->output_labels_)
		{

		}
	}

	template <typename Dtype>
	void VideoDataLayer<Dtype>::ShuffleVideos(){
		caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	

	template <typename Dtype>
	void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
	{
			int num = 0;
			if (!ReadVideoToBlob(lines_id_, num, batch))
			{
				cout << "Cannot read video!" << endl;
				return;
			}
			if (this->output_labels_)
			{
				//batch->labels 赋值
			}
			lines_id_++;
			if (lines_id_ >= lines_.size())
			{
				lines_id_ = 0;
				if (this->layer_param_.video_data_param().shuffle())
				{
					ShuffleVideos();
				}
			}
	}

	template <typename Dtype>
	bool VideoDataLayer<Dtype>::ReadVideoToBlob(int lines_id, int& num,Batch<Dtype>* batch)
	{
		video& item = lines_[lines_id];
		
		cv::VideoCapture capture(item.filename);
		if (!capture.isOpened())
		{
			return false;
		}
		int frame_num = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
		int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
		num = (frame_num / (step_ +1)) + 1;

		int temp = 0;

		cv::Mat frame;
		int count = step_;
		if (new_height_)
		{
			height = new_height_;
		}
		if (new_width_)
		{
			width = new_width_;
		}
		vector<int> shape(4);
		shape[0] = num;
		shape[1] = 3;
		shape[2] =  height;
		shape[3] =  width;
		this->prefetch_data_.Reshape(shape);
		 
		Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
		int f = 0;
		//降采样视屏
		while (true)
		{
			capture >> frame;
			f++;
			count++;
			if (frame.empty())
			{
				break;
			}
			if (count > step_)
			{
				cv::resize(frame, frame, cv::Size(width, height));
				for (int c = 0; c < frame.channels(); c++)
				{
					for (int h = 0; h < frame.cols; h++)
					{
						for (int w = 0; w < frame.rows; w++)
						{
							shape[0] = temp;
							shape[1] = c;
							shape[2] = h;
							shape[3] = w;
							prefetch_data[this->prefetch_data_.offset(shape)] = static_cast<Dtype>(frame.at<cv::Vec3b>(h, w)[c]);
						}
					}
				}

				temp++;
				count = 0;
			}
			
		}
		CHECK_EQ(num, temp) << "num does not equal to temp-------------------------video_data_layer" << endl;
		
		shape[0] = num;
		shape[1] = 3;
		shape[2] = crop_size_ > 0 ? crop_size_: height;
		shape[3] = crop_size_ > 0 ? crop_size_ : width;
		this->transformed_data_.Reshape(shape);
		batch->data_.Reshape(shape);
		Dtype* top_data = batch->data_.mutable_cpu_data();
		this->transformed_data_.set_cpu_data(top_data);
		this->data_transformer_->Transform(&prefetch_data_, &this->transformed_data_);
		cout << "num  = " << this->transformed_data_.num() << " channel =  " << this->transformed_data_.channels() 
			<< " height = " << this->transformed_data_.height() << " width = " << this->transformed_data_.width() << endl;
		return true;
	}

	INSTANTIATE_CLASS(VideoDataLayer);
	REGISTER_LAYER_CLASS(VideoData);
}
