#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  //对卷积核，步长，padding，权值和偏移设置
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //reshape top的形状
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  //caffe中，为了减少计算量，卷积的实现使用矩阵相乘的方法一次性计算出来，
  //具体的就是把卷积核展开排成一个矩阵，把对应要卷积的部分数据也展开成另外一个矩阵,然后相乘以便一次性算出来
  //http://blog.csdn.net/buyi_shizi/article/details/51504276
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  //
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;			//卷积核大小,Blob的shape存的是卷积核的维度大小(2维或3维),Blob的data存的是卷积核的h、w(3维的话还要多存一个d)
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;				//步长
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;					//输入填补
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;				//卷积核的拓展
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;			//输入和c、h、w形状
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;		//列缓存的shape，
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;		//bottom的形状

  int num_spatial_axes_;		//卷积核维数
  int bottom_dim_;				//bottom的c*h*w
  int top_dim_;					//top的c*h*w

  int channel_axis_;			//通道在什么轴				
  int num_;						//bottom的num
  int channels_;				//通道大小
  int group_;					//组
  int out_spatial_dim_;			//输出的h*w
  int weight_offset_;			//权值的总偏移量
  int num_output_;				//卷积后的输出数量
  bool bias_term_;					//是否需要bias
  bool is_1x1_;						//是否为1x1卷积，步长为1,没有padding
  bool force_nd_im2col_;			//强制n维卷积

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;				//输出的通道数(输出的特征数量)
  int conv_in_channels_;				//输入的通道数
  int conv_out_spatial_dim_;			//top的h*w
  int kernel_dim_;						//卷积核 c*h*w
  int col_offset_;						//列偏移
  int output_offset_;					//输出偏移

  Blob<Dtype> col_buffer_;				//列缓存
  Blob<Dtype> bias_multiplier_;			//偏移的乘数
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
