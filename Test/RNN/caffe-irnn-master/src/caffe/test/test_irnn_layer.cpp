#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/irnn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class IRNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  IRNNLayerTest()
      : blob_bottom_(new Blob<Dtype>(12, 3, 2, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~IRNNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IRNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(IRNNLayerTest, TestGradientDefault) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    IRNNParameter* irnn_param =
        layer_param.mutable_irnn_param();
    irnn_param->set_num_output(5);
    irnn_param->mutable_weight_filler()->set_type("uniform");
    irnn_param->mutable_weight_filler()->set_min(-1);
    irnn_param->mutable_weight_filler()->set_max(1);
    irnn_param->mutable_bias_filler()->set_type("constant");
    irnn_param->mutable_bias_filler()->set_value(0);
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);

    IRNNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(IRNNLayerTest, TestGradientBatchDefault) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    IRNNParameter* irnn_param =
        layer_param.mutable_irnn_param();
    irnn_param->set_num_output(5);
    irnn_param->set_batch_size(3);
    irnn_param->mutable_weight_filler()->set_type("uniform");
    irnn_param->mutable_weight_filler()->set_min(-1);
    irnn_param->mutable_weight_filler()->set_max(1);
    irnn_param->mutable_bias_filler()->set_type("constant");
    irnn_param->mutable_bias_filler()->set_value(0);
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);

    IRNNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
}  // namespace caffe
