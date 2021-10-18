#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
template <typename T>
class Queue
{
 public:

  T pop()
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while ((queue_.empty()) && !free_)
    {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    return val;
  }

  uint pop(T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while ((queue_.empty()) && !free_)
    {
      cond_.wait(mlock);
    }
    if (queue_.empty())
      return -1;
    item = queue_.front();
    queue_.pop();
    return 0;
  }

  void push(const T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= size_)
    {
        queue_.pop();
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_all();
  }
  void setSize(int size)
  {
      size_ = size;
  }
  bool empty()
  {
      return queue_.empty();
  }
  bool free()
  {
    free_ = true;
    cond_.notify_all();
  }
  Queue()=default;
  Queue(const Queue&) = delete;            // disable copying
  Queue& operator=(const Queue&) = delete; // disable assignment

 private:
  uint size_ = 20;
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool free_ = false;
};
#endif // CONCURRENT_QUEUE_H