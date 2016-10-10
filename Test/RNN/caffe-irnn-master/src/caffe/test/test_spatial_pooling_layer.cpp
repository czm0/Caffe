#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SpatialPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SpatialPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        pos_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()){}
  virtual void SetUp() {
    blob_bottom_->Reshape(10, 4, 1, 1);
    // fill the values
    blob_bottom_->mutable_cpu_data()[0] = 1;
    blob_bottom_->mutable_cpu_data()[1] = 2;
    blob_bottom_->mutable_cpu_data()[2] = 3;
    blob_bottom_->mutable_cpu_data()[3] = 4;
    blob_bottom_->mutable_cpu_data()[4] = 2;
    blob_bottom_->mutable_cpu_data()[5] = 6;
    blob_bottom_->mutable_cpu_data()[6] = 7;
    blob_bottom_->mutable_cpu_data()[7] = 8;
    blob_bottom_->mutable_cpu_data()[8] = 9;
    blob_bottom_->mutable_cpu_data()[9] = 10;
    blob_bottom_->mutable_cpu_data()[10] = 11;
    blob_bottom_->mutable_cpu_data()[11] = 12;
    blob_bottom_->mutable_cpu_data()[12] = 10;
    blob_bottom_->mutable_cpu_data()[13] = 8;
    blob_bottom_->mutable_cpu_data()[14] = 6;
    blob_bottom_->mutable_cpu_data()[15] = 4;
    blob_bottom_->mutable_cpu_data()[16] = 1;
    blob_bottom_->mutable_cpu_data()[17] = 3;
    blob_bottom_->mutable_cpu_data()[18] = 5;
    blob_bottom_->mutable_cpu_data()[19] = 7;
    blob_bottom_->mutable_cpu_data()[20] = 2;
    blob_bottom_->mutable_cpu_data()[21] = 4;
    blob_bottom_->mutable_cpu_data()[22] = 6;
    blob_bottom_->mutable_cpu_data()[23] = 8;
    blob_bottom_->mutable_cpu_data()[24] = 2;
	blob_bottom_->mutable_cpu_data()[25] = 2;
	blob_bottom_->mutable_cpu_data()[26] = 3;
	blob_bottom_->mutable_cpu_data()[27] = 9;
	blob_bottom_->mutable_cpu_data()[28] = 8;
	blob_bottom_->mutable_cpu_data()[29] = 7;
	blob_bottom_->mutable_cpu_data()[30] = 5;
	blob_bottom_->mutable_cpu_data()[31] = 1;
	blob_bottom_->mutable_cpu_data()[32] = 4;
	blob_bottom_->mutable_cpu_data()[33] = 6;
	blob_bottom_->mutable_cpu_data()[34] = 1;
	blob_bottom_->mutable_cpu_data()[35] = 3;
	blob_bottom_->mutable_cpu_data()[36] = 5;
	blob_bottom_->mutable_cpu_data()[37] = 9;
	blob_bottom_->mutable_cpu_data()[38] = 2;
	blob_bottom_->mutable_cpu_data()[39] = 10;


    pos_bottom_->Reshape(1,11,3,1);
    pos_bottom_->mutable_cpu_data()[0] = 100;
    pos_bottom_->mutable_cpu_data()[1] = 100;
    pos_bottom_->mutable_cpu_data()[2] = 0;
    pos_bottom_->mutable_cpu_data()[3] = 140;
    pos_bottom_->mutable_cpu_data()[4] = 60;
    pos_bottom_->mutable_cpu_data()[5] = 1;
    pos_bottom_->mutable_cpu_data()[6] = 80;
    pos_bottom_->mutable_cpu_data()[7] = 300;
    pos_bottom_->mutable_cpu_data()[8] = 2;
    pos_bottom_->mutable_cpu_data()[9] = 160;
    pos_bottom_->mutable_cpu_data()[10] = 280;
    pos_bottom_->mutable_cpu_data()[11] = 0;
    pos_bottom_->mutable_cpu_data()[12] = 200;
    pos_bottom_->mutable_cpu_data()[13] = 140;
    pos_bottom_->mutable_cpu_data()[14] = 4;
    pos_bottom_->mutable_cpu_data()[15] = 20;
    pos_bottom_->mutable_cpu_data()[16] = 80;
    pos_bottom_->mutable_cpu_data()[17] = 5;
    pos_bottom_->mutable_cpu_data()[18] = 140;
    pos_bottom_->mutable_cpu_data()[19] = 200;
    pos_bottom_->mutable_cpu_data()[20] = 1;
    pos_bottom_->mutable_cpu_data()[21] = 30;
    pos_bottom_->mutable_cpu_data()[22] = 180;
    pos_bottom_->mutable_cpu_data()[23] = 4;
    pos_bottom_->mutable_cpu_data()[24] = 80;
    pos_bottom_->mutable_cpu_data()[25] = 140;
    pos_bottom_->mutable_cpu_data()[26] = 2;
    pos_bottom_->mutable_cpu_data()[27] = 200;
    pos_bottom_->mutable_cpu_data()[28] = 20;
    pos_bottom_->mutable_cpu_data()[29] = 0;
    pos_bottom_->mutable_cpu_data()[30] = 240;
    pos_bottom_->mutable_cpu_data()[31] = 320;
    pos_bottom_->mutable_cpu_data()[32] = 6;


    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(pos_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialPoolingLayerTest() {
    delete blob_bottom_;
    delete pos_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const pos_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(SpatialPoolingLayerTest, TestDtypesAndDevices);


TYPED_TEST(SpatialPoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpatialPoolingParameter* pooling_param = layer_param.mutable_spatial_pooling_param();
  pooling_param->set_num_h(2);
  pooling_param->set_num_w(2);
  pooling_param->set_num_t(2);

  SpatialPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_,0);
}


}  // namespace caffe
