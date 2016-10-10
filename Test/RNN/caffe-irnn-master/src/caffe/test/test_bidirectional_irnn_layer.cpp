#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bidirectional_irnn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class BidirectionalIRNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BidirectionalIRNNLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 4, 3, 6)),
        ftop_(new Blob<Dtype>()),btop_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    Dtype ibottom[]={-0.0694,	0.0379,	0.0209,	-0.0614,	-0.0201,	-0.0046,
    		-0.0001,	-0.0970,	-0.0182,	0.0242,	-0.0835,	-0.0407,
    		0.0443,	-0.0801,	0.0169,	-0.0044,	0.0905,	0.0360,
    		-0.0781,	-0.0409,	0.0365,	-0.0493,	-0.0833,	-0.0666,
    		-0.0396,	0.0723,	-0.0424,	-0.0882,	0.0375,	-0.0558,
    		0.0296,	0.0169,	0.0092,	0.0667,	-0.0198,	0.0132,
    		0.0594,	0.0855,	-0.0196,	-0.0548,	-0.0486,	-0.0528,
    		0.0917,	-0.0908,	-0.0210,	-0.0478,	-0.0699,	-0.0278,
    		0.0189,	-0.0533,	0.0497,	0.0941,	-0.0379,	-0.0233,
    		0.0171,	0.0211,	-0.0214,	0.0951,	0.0795,	-0.0739,
    		-0.0024,	-0.0263,	0.0854,	0.0428,	0.0011,	-0.0012,
    		0.0892,	0.0106,	0.0089,	0.0991,	0.0569,	0.0162
    };
    caffe_copy(72, ibottom, this->blob_bottom_->mutable_cpu_data());
    blob_top_vec_.push_back(ftop_);
    blob_top_vec_.push_back(btop_);
  }
  virtual ~BidirectionalIRNNLayerTest() { delete blob_bottom_; delete ftop_;delete btop_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const ftop_;
  Blob<Dtype>* const btop_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BidirectionalIRNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(BidirectionalIRNNLayerTest, TestGradientDefault) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    BidirectionalIRNNParameter* irnn_param =
        layer_param.mutable_bidirectional_irnn_param();
    irnn_param->set_num_output(5);
    irnn_param->set_trajectory_len(4);
    irnn_param->mutable_weight_filler()->set_type("uniform");
    irnn_param->mutable_weight_filler()->set_min(-1);
    irnn_param->mutable_weight_filler()->set_max(1);
    irnn_param->mutable_bias_filler()->set_type("constant");
    irnn_param->mutable_bias_filler()->set_value(0);
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);

    BidirectionalIRNNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
