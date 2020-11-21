#ifndef TEMPLATE_MATCHER_H
#define TEMPLATE_MATCHER_H

#include <QDir>
#include <QtGlobal>
#include <QThread>

#include "opencv2/opencv.hpp"

#include <mutex>

#include "sync_queue.h"

/**
 * @brief Struct used to store a frame and its id.
 */
struct Frame
{
    int id;
    cv::Mat* mat;
};

/**
 * @brief This object will process a series of frame
 * to show them on screen, will find a template on them,
 * will identify the template with a rectangle on the frame
 * and will write the resulting frame in a file.
 *
 * This object creates a producer/consumers mechanism
 * used to show and process the frames. The "producer"
 * is the processFrame method, that should be called
 * in the main thread loop. The "consumers" are the threads
 * created in this class to match the frames in background.
 *
 * The matching method used is TM_SQDIFF.
 */
class template_matcher
{
 public:
    static constexpr auto OUT_FILES_DIR          {"frames/"};
    static constexpr auto OUT_FILE_FORMAT        {"frame%04d.png"};

    /**
     * @brief Ctor
     * @param templ template used to for the match
     * @param mask Mask of the template
     * @param thread_count Number of thread to use for the consumers
     * @param window The CV window name on which we will show the frames
     */
    template_matcher(cv::Mat& templ, cv::Mat mask, size_t thread_count, const char* window)
        : templ(templ), mask(mask), window(window), thread_count(thread_count)
    {
        Q_CHECK_PTR(window);

        create_frames_dir();
        reset_counters();
        launch_consumers();
    }

    /**
     * @brief Reset the match counters.
     */
    void reset_counters(void)
    {
        frame_to_match_counter = 0;
        frame_matched_counter = 0;
    }

    /**
     * @brief Check if the consumer threads are still running.
     * Thread-safe.
     * @return True if consumers are running, false otherwise.
     */
    bool are_consumers_running(void)
    {
        std::unique_lock<std::mutex> mlock(mutex);
        return (! consumers.empty());
    }

    /**
     * @brief Dtor. Will call the kill_consumers method.
     */
    ~template_matcher()
    {
        kill_consumers();
    }

    /**
     * @return The number of frames to match.
     */
    int get_frame_to_match_counter() const
    {
        return frame_to_match_counter;
    }

    /**
     * @return The number of frames already matched.
     */
    int get_frame_matched_counter() const
    {
        return frame_matched_counter;
    }

    void processFrame(cv::Mat& mat);
    void kill_consumers(void);

private:
    sync_queue<Frame> sq;

    cv::Mat templ;
    cv::Mat mask;

    std::vector<std::thread> consumers;
    std::mutex mutex;
    std::atomic_int frame_matched_counter;

    const char* window;
    size_t thread_count;
    int frame_to_match_counter;

    /**
     * @brief Create the ouput dir if it does not exist
     */
    void create_frames_dir(void)
    {
        if(! QDir(OUT_FILES_DIR).exists())
        {
            QDir().mkdir(OUT_FILES_DIR);
        }
    }

    void matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, int match_method, bool use_mask);
    void launch_consumers(void);
};

#endif // TEMPLATE_MATCHER_H
