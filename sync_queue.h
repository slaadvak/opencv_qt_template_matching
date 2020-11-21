#ifndef SYNC_QUEUE_H
#define SYNC_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief This object encapsulates a std::queue to make it thread-safe.
 * @see std::queue
 */
template <typename T>
class sync_queue
{
 public:

    /**
     * @brief pop Removes the next element in the queue.
     * Will call queue::front then queue::pop.
     * @param item A reference that will point to the retrieved element.
     * @see stq::queue::front
     * @see stq::queue::pop
     */
    void pop(T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex);
        while (queue.empty())
        {
            cv.wait(mlock);
        }
        item = queue.front();
        queue.pop();
        return;
    }

    /**
     * @brief push Push an element into the queue.
     * Will call queue::push.
     * @param item The item to push.
     * @see std::queue::push
     */
    void push(const T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex);
        queue.push(item);
        mlock.unlock();
        cv.notify_one();
    }

 private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;
};

#endif // SYNC_QUEUE_H
