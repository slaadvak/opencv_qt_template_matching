#include "template_matcher.h"

#include <QElapsedTimer>
#include <QtDebug>

/**
 * @brief Matches a template against a frame and then draw a rectangle around it on the frame.
 * @param frame The frame to find the template on
 * @param templ The template to find on the frame
 * @param mask The mask to use with the template ir necessary. Might be an empty cv::Mat otherwise.
 * @param match_method The method passed to matchTemplate.
 * @param use_mask True if the match_method requires a mask, false otherwise.
 * @see cv::matchTemplate
 */
void template_matcher::matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, int match_method, bool use_mask)
{
  int result_cols =  frame.cols - templ.cols + 1;
  int result_rows = frame.rows - templ.rows + 1;

  cv::Mat result( result_rows, result_cols, CV_32FC1 );
  bool method_accepts_mask = (cv::TM_SQDIFF == match_method || match_method == cv::TM_CCORR_NORMED);
  if (use_mask && method_accepts_mask)
  {
      matchTemplate( frame, templ, result, match_method, mask);
  }
  else
  {
      matchTemplate( frame, templ, result, match_method);
  }

  normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

  if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
  {
      matchLoc = minLoc;
  }
  else
  {
      matchLoc = maxLoc;
  }

  rectangle(frame, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar(255, 0, 0), 2, 8, 0 );

  return;
}

/**
 * @brief Method to show and push a frame inside the
 * template_matcher for later process
 * @param mat The mat to process
 */
void template_matcher::processFrame(cv::Mat& mat)
{
    const int sleep_period_ms = 17;

    QElapsedTimer timer;
    timer.start();

    // your implementation of the template matching goes here
    //=======================================================

    // This new object will be deleted by the consumer thread
    cv::Mat* copy = new cv::Mat();
    mat.copyTo(*copy);

    sq.push({frame_to_match_counter, copy});

    imshow(window, mat);
    cv::waitKey(1);

    frame_to_match_counter += 1;
    qDebug() << "processFrame elapsed" << timer.elapsed() << "ms";

    //=======================================================

    QThread::msleep(std::max(0, static_cast<int>(sleep_period_ms - timer.elapsed())));
}

/**
 * @brief Send a signal to kill all the consumers
 * then join on them waiting for them to stop.
 *
 * Thread-safe.
 */
void template_matcher::kill_consumers(void)
{
    // Send a special Frame that will stop the consumers
    for(size_t i = 0; i < consumers.size(); i++)
    {
        sq.push({0, nullptr});
    }

    // Wait for each consumer to stop
    for (auto& consumer : consumers)
    {
        consumer.join();
    }

    {
        std::unique_lock<std::mutex> mlock(mutex);
        consumers.clear();
    }
}

/**
 * @brief Launch the consumer threads used to process
 * the images
 */
void template_matcher::launch_consumers()
{
    // Main loop to be run inside the consumer threads
    auto thread_lambda = [this](int thread_id)
    {
        qDebug() << "Starting T#" << thread_id;

        while(true)
        {
            // Retrieve frame
            Frame f{};
            sq.pop(f);

            // Are we asked to stop ?
            if(f.mat == nullptr)
            {
                qDebug() << "Stopping T#" << thread_id;
                break;
            }

            qDebug() << "T#" << thread_id << "on frame #" << f.id;

            matching_method(*f.mat, this->templ, this->mask, cv::TM_CCORR_NORMED, true);

            // Create filename
            auto size = std::snprintf(nullptr, 0, OUT_FILE_FORMAT, f.id);
            std::string filename(size + 1, '\0');
            std::sprintf(&filename[0], OUT_FILE_FORMAT, f.id);

            filename = OUT_FILES_DIR + filename;

            // Write file
            if(! cv::imwrite(filename, *f.mat))
            {
                qCritical() << "Error writing file " << filename.c_str();
            }

            frame_matched_counter += 1;

            // Delete bits array created in enqueue_frame
            delete f.mat;
        }
    };

    // Create and launch the threads
    for(size_t i = 0; i < thread_count; i++)
    {
        std::thread t(thread_lambda, i);

        consumers.push_back(std::move(t));
    }
}


