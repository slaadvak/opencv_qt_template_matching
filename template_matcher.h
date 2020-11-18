#ifndef TEMPLATE_MATCHER_H
#define TEMPLATE_MATCHER_H

#include <QDebug>
#include <QElapsedTimer>
#include <QThread>

#include "opencv2/opencv.hpp"

#include "sync_queue.h"

struct Frame
{
    int id;
    cv::Mat* mat;
};

void matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, cv::Mat& img_display, int match_method, bool use_mask = true);

void matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, cv::Mat& img_display, int match_method, bool use_mask)
{
  frame.copyTo( img_display );

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

  rectangle( img_display, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );

  return;
}

class template_matcher
{
 public:
    static constexpr auto OUT_FILES_DIR          {"frames/"};
    static constexpr auto OUT_FILE_FORMAT        {"frame%04d.png"};

    template_matcher(cv::Mat& templ, cv::Mat mask, size_t thread_count)
        : templ(templ), mask(mask), frame_count(0)
    {
        for(size_t i = 0; i < thread_count; i++)
        {
            std::thread t([this](int thread_id)
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

                    // TODO : get rid of img_display, reuse f.mat instead!!!
                    cv::Mat img_display;
                    matching_method(*f.mat, this->templ, this->mask, img_display, cv::TM_SQDIFF, true);

                    // Create filename
                    auto size = std::snprintf(nullptr, 0, OUT_FILE_FORMAT, f.id);
                    std::string filename(size + 1, '\0');
                    std::sprintf(&filename[0], OUT_FILE_FORMAT, f.id);

                    filename = OUT_FILES_DIR + filename;

                    // Write file
                    if(! cv::imwrite(filename, img_display))
                    {
                        qDebug() << "Error writing file " << filename.c_str();
                    }

                    // Delete bits array created in enqueue_frame
                    delete f.mat;
                }
            }, i);

            consumers.push_back(std::move(t));
        }
    }

    void processFrame(cv::Mat& mat)
    {
        const int sleep_period_ms = 17;

        QElapsedTimer timer;
        timer.start();

        // your implementation of the template matching goes here
        //=======================================================
        imshow("Frame", mat);
        cv::waitKey(1);

        // This new object will be deleted by the consumer thread
        cv::Mat* copy = new cv::Mat(mat.size(), mat.type());
        mat.copyTo(*copy);

        sq.push({frame_count, copy});

        frame_count += 1;
        //=======================================================

        qDebug() << "processFrame elapsed" << timer.elapsed() << "ms";

        ulong time_diff = sleep_period_ms - timer.elapsed();
        time_diff = MIN(0, time_diff);
        QThread::msleep(time_diff);
    }

    ~template_matcher()
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
    }

  private:
    cv::Mat templ;
    cv::Mat mask;
    sync_queue<Frame> sq;
    std::vector<std::thread> consumers;
    int frame_count;

};

#endif // TEMPLATE_MATCHER_H
