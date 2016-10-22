#pragma once
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/layers/base_data_layer.hpp"


namespace caffe
{
	struct video
	{
		string filename;
		int label;
	};

	template <typename Dtype>
	class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit VideoDataLayer(const LayerParameter& param): BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~VideoDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }


	protected:

		

		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual void ShuffleVideos();
		virtual void load_batch(Batch<Dtype>* batch);


		vector<video> lines_;
		int lines_id_;
		Blob<Dtype> prefetch_data_;
		std::ifstream infile;
		int step_;
		int new_height_;
		int new_width_;
		int crop_size_;

	private:
		bool ReadVideoToBlob(int lines_id, int& num, Batch<Dtype>* batch);
		
	};
}