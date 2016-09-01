#include <boost/thread.hpp>
#include <string>

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

	//当生产者和消费者线程需要对同一资源进行操作的时候需要互斥锁。
	/*
	*阻塞条件
	* 1、缓冲区空，此时消费者不能消费，拒绝pop操作之后，可以交出CPU控制权。
	* 2、缓冲区满，此时生产者不能生产，拒绝push操作之后，可以交出CPU控制权。
	*对于阻塞的线程，该线程不能主动激活，需要由其他线程激活
	*/

template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;						//互斥锁
  boost::condition_variable condition_;				//blocking阻塞，与互斥不同，互斥会将多个线程对同一个资源的异步并行操作，拉成一个串行执行队列，串行等待执行。而blocking则是将线程休眠，CPU会暂时放弃对其控制。
};

template<typename T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);		//scoped_lock能锁定局部对象，在作用域结束之后立刻解锁，可以不需要我们自己手动调用unlock
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();				//激活一个线程
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    sync_->condition_.wait(lock);		//使用当前mutex为标记，交出cpu控制权
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

//实例化类
template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;
template class BlockingQueue<Datum*>;
template class BlockingQueue<shared_ptr<DataReader::QueuePair> >;
template class BlockingQueue<P2PSync<float>*>;
template class BlockingQueue<P2PSync<double>*>;

}  // namespace caffe
