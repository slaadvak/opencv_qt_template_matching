#ifndef SYNC_QUEUE_H
#define SYNC_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class sync_queue
{
 public:

  bool pop(T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex);
    if(queue.empty())
    {
        return false;
    }
//    while (queue.empty())
//    {
//      cv.wait(mlock);
//    }
    item = queue.front();
    queue.pop();
    return true;
  }

  void push(const T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex);
    queue.push(item);
    mlock.unlock();
    //cv.notify_one();
  }

 private:
  std::queue<T> queue;
  std::mutex mutex;
  //std::condition_variable cv;
};

#endif // SYNC_QUEUE_H
