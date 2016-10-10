#ifndef CAFFE_RENET_LSTM_LAYER_HPP_
#define CAFFE_RENET_LSTM_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/* implement the ReNet layer in paper
 * "ReNet: A Recurrent Neural Network Based
 * Alternative to Convolutional Networks"
 * URL: http://arxiv.org/abs/1505.00393
 * */
template<typename Dtype>
class ReNetLSTMLayer: public Layer<Dtype> {
public:
  explicit ReNetLSTMLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const {
    return "ReNetLSTM";
  }
  /* @brief Input is a blob of shape (num, channels, height, width)
   * */
  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }
  /* @brief Output a single blob consisting of 2 stacked layers of hidden states
   * */
  virtual inline int ExactNumTopBlobs() const {
    return 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //把图片和H填入rnn中，输入为[X H]
  void Fill_X_H_Data_cpu(int dir_num, int step_id, int step_start,
      Blob<Dtype>* bottom);
  void ComputeCellData_cpu(int dir_num, int step_id, int step_start,
      Blob<Dtype>* top);
  void ComputeCellData_gpu(int dir_num, int step_id, int step_start,
      Blob<Dtype>* top);
  void FillHiddenDiff_cpu(int dir_num, int step_id, int step_end,
      Blob<Dtype>* top);
  void ComputeCellDiff_cpu(int dir_num, int step_id, int step_start,
      int step_end);
  void ComputeCellDiff_gpu(int dir_num, int step_id, int step_start,
      int step_end);
  void Compute_X_H_Diff_cpu(int dir_num, int step_id, int step_start,
      Blob<Dtype>* bottom);
  void Compute_X_H_Diff_gpu(int dir_num, int step_id, int step_start,
      Blob<Dtype>* bottom);
  void ComputeParamDiff_cpu(int dir_num, int step_id, int step_start);
  void ComputeParamDiff_gpu(int dir_num, int step_id, int step_start);

  bool peephole_;				//是否让门接收cell
  int patch_h_;
  int patch_w_;
  int num_output_;						//输出大小
  ReNetLSTMParameter::Direction dir_;			//x方向还是y方向

  int num_;						//样本数
  int channels_;				//通道数
  int patch_ny_;				//y方向patch数量
  int patch_nx_;				//x方向patch数量
  int patch_dim_;				//channels_ * patch_h_ * patch_w_;
  int num_blobs_per_dir_;		//每个方向的权值和偏移
  int num_RNN_;					//rnn数量
  int num_steps_;				//每个rnn的输出数量

  Blob<Dtype> bias_multiplier_;

  vector<shared_ptr<Blob<Dtype> > > X_H_data_, X_H_diff_;			//[X H]
  vector<shared_ptr<Blob<Dtype> > > gi_data_, gi_diff_, gi_next_diff_;
  vector<shared_ptr<Blob<Dtype> > > ci_data_, ci_diff_;
  vector<shared_ptr<Blob<Dtype> > > go_data_, go_diff_;
  vector<shared_ptr<Blob<Dtype> > > gf_data_, gf_diff_, gf_next_diff_;
  vector<shared_ptr<Blob<Dtype> > > cstate_data_, cstate_diff_,
      cstate_next_diff_;
  vector<shared_ptr<Blob<Dtype> > > hidden_data_, hidden_diff_;
};
} // namespace caffe

#endif // #ifndef CAFFE_RENET_LSTM_LAYER_HPP_
