#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/action_irnn_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype relu(Dtype x) {
  return (x >= Dtype(0)) ? x : 0;
}

/*template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}*/


template <typename Dtype>
void ActionIRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  H_ = this->layer_param_.action_irnn_param().num_output(); // number of hidden units
  //I_ = bottom[0]->count() / bottom[0]->num(); // input dimension
  I_ = bottom[0]->width();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.action_irnn_param().weight_filler()));

    // input-to-hidden weights
    // Intialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(H_);
    weight_shape.push_back(I_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    weight_shape.clear();
    weight_shape.push_back(H_);
    weight_shape.push_back(H_);
    float factor = this->layer_param_.action_irnn_param().hidden_init_factor();
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    Dtype* hh_weight_data = this->blobs_[1]->mutable_cpu_data();
    caffe_set(this->blobs_[1]->count(), Dtype(0.), hh_weight_data);

    for(int i=0; i<H_; i++){
      hh_weight_data[i*H_+i]=Dtype(factor);
    }

    // If necessary, intiialize and fill the bias term
    vector<int> bias_shape(1, H_);
    this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.action_irnn_param().bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ActionIRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  T_ = this->layer_param_.action_irnn_param().trajectory_len(); // trajectory_len
  //N_ = bottom[0]->num() / T_; // length of sequence
  N_ = bottom[0]->height();
  vector<int> original_top_shape;
  original_top_shape.push_back(N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  // Top initialization
  vector<int> top_shape;
  top_shape.push_back(T_);
  top_shape.push_back(N_);
  top_shape.push_back(H_);
  pre_gate_.Reshape(top_shape);
  top_.Reshape(top_shape);
 // top_.ShareData(*top[0]);
 // top_.ShareDiff(*top[0]);

  // Set up the bias multiplier
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1),
    bias_multiplier_.mutable_cpu_data());

  vector<int> hidden_shape;
  hidden_shape.push_back(N_);
  hidden_shape.push_back(H_);
  h_0_.Reshape(hidden_shape);
  h_T_.Reshape(hidden_shape);
  h_to_h_.Reshape(hidden_shape);
  h_to_gate_.Reshape(hidden_shape);
}

template <typename Dtype>
void ActionIRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight_i = this->blobs_[0]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();
  const Dtype* bias = this->blobs_[2]->cpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
  Dtype* h_to_gate = h_to_gate_.mutable_cpu_data();

  // Initialize previous state
  caffe_set(h_0_.count(), Dtype(0.), h_0_.mutable_cpu_data());


  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, H_, I_, Dtype(1.),
      bottom_data, weight_i, Dtype(0.), pre_gate_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, H_, 1, Dtype(1.),
      bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + top_.offset(t);
    Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    Dtype* h_to_gate_t=h_to_gate;
    const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();

    // Hidden-to-hidden propagation
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, H_, H_, Dtype(1.),
        h_t_1, weight_h, Dtype(0.), h_to_gate_t);

    for (int n = 0; n < N_; ++n) {
      const bool cont = t > 0;
      if (cont) {
        caffe_add(H_, pre_gate_t, h_to_gate_t, pre_gate_t);
      }
      for (int d = 0; d < H_; ++d) {
        // Apply nonlinearity
        h_t[d] = relu(pre_gate_t[d]);
    	//h_t[d] = sigmoid(pre_gate_t[d]);
      }

      h_t += H_;
      pre_gate_t += H_;
      h_to_gate_t += H_;
    }
  }

  caffe_copy(N_*H_, top_data + top_.offset(T_-1), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void ActionIRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.cpu_data();
  Dtype* top_diff = top_.mutable_cpu_diff();
  caffe_set(top_.count(), Dtype(0), top_diff);
  caffe_copy(N_*H_, top[0]->mutable_cpu_diff(), top_diff + top_.offset(T_-1));

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const Dtype* weight_i = this->blobs_[0]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();

  const Dtype* pre_gate_data = pre_gate_.cpu_data();
  Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();

  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + top_.offset(t);
    Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
    Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_cpu_diff();
    const Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);

    for (int n = 0; n < N_; ++n) {
      for (int d = 0; d < H_; ++d) {
        pre_gate_diff_t[d] = dh_t[d] * ( (pre_gate_t[d] > Dtype(0) )? 1 : 0);
    	//pre_gate_diff_t[d] = dh_t[d] * (1-sigmoid(pre_gate_t[d]))* sigmoid(pre_gate_t[d]);
      }

      dh_t += H_;
      pre_gate_t += H_;
      pre_gate_diff_t += H_;
    }

    // Backprop output errors to the previous time step
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, H_,
        Dtype(1.), pre_gate_diff + pre_gate_.offset(t),
        weight_h, Dtype(0.), h_to_h_.mutable_cpu_data());
    for (int n = 0; n < N_; ++n) {
      const bool cont = t > 0;
      const Dtype* h_to_h = h_to_h_.cpu_data() + h_to_h_.offset(n);
      if (cont) {
        caffe_add(H_, dh_t_1, h_to_h, dh_t_1);
      }
      dh_t_1 += H_;
    }
  }

  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, H_, I_, T_*N_, Dtype(1.),
        pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_cpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, H_, H_, (T_-1)*N_, Dtype(1.),
        pre_gate_diff + pre_gate_.offset(1), top_data,
        Dtype(1.), this->blobs_[1]->mutable_cpu_diff());

    // Add Gradient from previous time-step
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, H_, H_, 1, Dtype(1.),
        pre_gate_diff, h_0_.cpu_data(),
        Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[2]) {
    // Gradient w.r.t. bias
    caffe_cpu_gemv(CblasTrans, T_*N_, H_, Dtype(1.), pre_gate_diff,
        bias_multiplier_.cpu_data(), Dtype(1.),
        this->blobs_[2]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, H_, Dtype(1.),
        pre_gate_diff, weight_i, Dtype(0.), bottom_diff);
  }

}

#ifdef CPU_ONLY
STUB_GPU(ActionIRNNLayer);
#endif

INSTANTIATE_CLASS(ActionIRNNLayer);
REGISTER_LAYER_CLASS(ActionIRNN);

}  // namespace caffe
