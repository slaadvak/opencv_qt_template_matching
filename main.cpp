#include "sync_queue.h"

#include <QDebug>
#include <QElapsedTimer>
#include <QThread>

#include "opencv2/opencv.hpp"

#include <filesystem>

void matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& result, cv::Mat& mask, cv::Mat& img_display, int match_method, bool use_mask = false);

std::string& type2str(int type, std::string& r) {

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

//void processFrame(cv::Mat& mat)
//{
//  const int sleep_period_ms = 17;

//  QElapsedTimer timer;
//  timer.start();

//  // your implementation of the template matching goes here

//  ulong time_diff = sleep_period_ms - timer.elapsed();
//  time_diff = MIN(0, time_diff);
//  QThread::msleep(time_diff);
//}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    qDebug() << "CWD:" << std::filesystem::current_path().c_str();
    qDebug() << cv::getBuildInformation().c_str();

    cv::VideoCapture cap("/home/vm/imagia/field.mp4");
    cv::Mat templ = cv::imread("/home/vm/imagia/template.png");
    cv::Mat full_templ = cv::imread("/home/vm/imagia/template.png", cv::IMREAD_UNCHANGED);


    if(templ.empty())
    {
        qDebug() << "Cannot load template!";
        return -1;
    }

//    qDebug() << "Template channels:" << templ.channels();
//    imshow("templ", templ);

    cv::Mat mask( full_templ.rows, full_templ.cols, CV_8UC1 );
    cv::extractChannel(full_templ, mask, 3);
//    imshow("mask", mask);

    // Check if camera opened successfully
    if(!cap.isOpened())
    {
        qDebug() << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1)
    {
        static uint counter = 0;

        cv::Mat frame;

        // Capture frame-by-frame
        cap >> frame;

        if(frame.empty())
        {
          break;
        }

        qDebug() << "Frame" << counter;
        counter += 1;

        QElapsedTimer timer;
        timer.start();

        if(counter == 1)
        {
            std::string type;
            qDebug() << "Type:" << type2str(frame.type(), type).c_str()
                     << frame.cols << "x" << frame.rows;
            qDebug() << "Continuous:" << frame.isContinuous();
        }

        cv::Mat frame_gray;
//        cv::cvtColor(frame, frame_gray, cv::COLOR_YUV2GRAY_420);

        qDebug() << "Gray channels:" << frame_gray.channels();

        // Display the resulting frame
        imshow("Frame", frame);

        qDebug() << "elapsed1=" << timer.elapsed() << "ms";

        cv::Mat result;
        cv::Mat img_display;
        matching_method(frame, templ, mask, result, img_display, cv::TM_SQDIFF, true);

        qDebug() << "elapsed2=" << timer.elapsed() << "ms";

        imshow("result", result);
        imshow("img_display", img_display);

        cv::waitKey(1);

        qDebug() << "elapsed3=" << timer.elapsed() << "ms";

//        processFrame(frame);

        qDebug() << "elapsed4=" << timer.elapsed() << "ms";

        qDebug() << "==============================";
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}


void matching_method( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, cv::Mat& result, cv::Mat& img_display, int match_method, bool use_mask)
{
  frame.copyTo( img_display );

  int result_cols =  frame.cols - templ.cols + 1;
  int result_rows = frame.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );
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
  rectangle( result, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );

  return;
}
