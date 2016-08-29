#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
//CUDA使用memory pinned技术,使得分配的内存不会被释放，这样可以加速CPU和GPU数据的传输速度
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));		//GPU下使用cuda分配内存
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);			//如果只是用cpu则用c的malloc分配内存
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}
//释放内存
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  //构造函数，初始化参数
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();		//析构函数，调用CaffeFreeHost释放内存


  const void* cpu_data();	//获得cpu数据指针
  void set_cpu_data(void* data);		//设置cpu共享数据
  const void* gpu_data();	//获得gpu数据指针
  void set_gpu_data(void* data);    //设置gpu共享数据
  void* mutable_cpu_data(); //获得可更改的cpu数据指针
  void* mutable_gpu_data(); //获得可更改的gpu数据指针
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

  //异步同步数据
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();		//状态转移函数
  void to_gpu();		//状态转移函数
  void* cpu_ptr_;		//cpu内存指针
  void* gpu_ptr_;		//gpu显存指针
  size_t size_;		//内存大小
  SyncedHead head_;	//数据状态
  bool own_cpu_data_;		//共享标记，是否使用的是自己的cpu数据
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;		//共享标记，是否使用的是自己的gpu数据
  int gpu_device_;			//gpu设备

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
