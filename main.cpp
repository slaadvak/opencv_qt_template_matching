#include "sync_queue.h"
#include "template_matcher.h"

#include <QDebug>
#include <QElapsedTimer>
#include <QThread>

#include "opencv2/opencv.hpp"

#include <filesystem>

void drawMatch(cv::Mat& templ, cv::Mat& frame, std::vector<cv::Point>& match_centers);
void matching_method3( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, std::vector<cv::Point>& results, int match_method, bool use_mask, float thres);
void matching_method2(cv::Mat& img, const cv::Mat& templ, const cv::Mat& mask, int match_method);

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

void main_loop(cv::VideoCapture& cap, cv::Mat& templ, cv::Mat& mask, int thread_count);

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    QThread::currentThread()->setPriority(QThread::HighPriority);

    qDebug() << "CWD:" << std::filesystem::current_path().c_str();
//    qDebug() << cv::getBuildInformation().c_str();

    cv::VideoCapture cap("/home/vm/imagia/field.mp4");
    cv::Mat templ = cv::imread("/home/vm/imagia/template.png");
    cv::Mat full_templ = cv::imread("/home/vm/imagia/template.png", cv::IMREAD_UNCHANGED);


    if(templ.empty())
    {
        qDebug() << "Cannot load template!";
        return -1;
    }


    cv::Mat mask( full_templ.rows, full_templ.cols, CV_8UC1 );
    cv::extractChannel(full_templ, mask, 3);

    // Check if video file opened successfully
    if(!cap.isOpened())
    {
        qDebug() << "Error opening video stream or file" << endl;
        return -1;
    }

    main_loop(cap, templ, mask, QThread::idealThreadCount() - 1);

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}

void main_loop(cv::VideoCapture& cap, cv::Mat& templ, cv::Mat& mask, int thread_count)
{
    template_matcher tm(templ, mask, thread_count);

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

        qDebug() << "Producing frame" << counter;
        counter += 1;

//        if(counter == 1)
//        {
//            std::string type;
//            qDebug() << "Type:" << type2str(frame.type(), type).c_str()
//                     << frame.cols << "x" << frame.rows;
//            qDebug() << "Continuous:" << frame.isContinuous();
//        }

        // Display the resulting frame

        tm.processFrame(frame);

        qDebug() << "==============================";
    }
}

void matching_method3( cv::Mat& frame, cv::Mat& templ, cv::Mat& mask, std::vector<cv::Point>& results, int match_method, bool use_mask, float thres)
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

  qDebug() << "result1=" << result.rows << "x" << result.cols;

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::Point matchLoc;

  if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
  {
      matchLoc = minLoc;
      threshold(result, result, 0.1, 1, cv::THRESH_BINARY_INV);
  }
  else
  {
      matchLoc = maxLoc;
      threshold(result, result, 0.9, 1, cv::THRESH_TOZERO);
  }

  qDebug() << "result2=" << result.rows << "x" << result.cols;

    maxVal = 1.f;
    while (maxVal > thres)
    {
       minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
       qDebug() << "resultx=" << result.rows << "x" << result.cols;
       if (maxVal > thres)
       {
//           rectangle(result,Point(maxLoc.x - frame.cols/2,maxLoc.y - frame.rows/2),Point(maxLoc.x + object.cols/2,maxLoc.y + object.rows/2),Scalar::all(0),-1);
           results.push_back(maxLoc);
       }
    }


//  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

//  if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
//  {
//      matchLoc = minLoc;
//  }
//  else
//  {
//      matchLoc = maxLoc;
//  }

//  rectangle( img_display, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );
//  rectangle( result, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );

  return;
}

using namespace cv;

void matching_method2(Mat& img, const Mat& templ, const Mat& mask, int match_method)
{
    /// Source image to display
    Mat img_display; Mat result;
//   if(img.channels()==3)
//        cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.copyTo( img_display );//for later show off

    /// Create the result matrix - shows template responces
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create( result_cols, result_rows, CV_32FC1 );

    /// Do the Matching and Normalize
    matchTemplate( img, templ, result, match_method, mask );
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal;
    Point minLoc; Point maxLoc;
    Point matchLoc;


    //in my variant we create general initially positive mask
    Mat general_mask=Mat::ones(result.rows,result.cols,CV_8UC1);

    for(int k=0;k<5;++k)// look for N=5 objects
    {
        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, general_mask);
        //just to visually observe centering I stay this part of code:
        result.at<float>(minLoc ) =1.0;//
        result.at<float>(maxLoc ) =0.0;//

        // For SQDIFF and SQDIFF_NORMED, the best matches are lower values.
         //For all the other methods, the higher the better
        if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
            matchLoc = minLoc;
        else
            matchLoc = maxLoc;
                                //koeffitient to control neiboring:
        //k_overlapping=1.- two neiboring selections can overlap half-body of     template
        //k_overlapping=2.- no overlapping,only border touching possible
        //k_overlapping>2.- distancing
        //0.< k_overlapping <1.-  selections can overlap more then half
        float k_overlapping=1.7f;//little overlapping is good for my task

        //create template size for masking objects, which have been found,
        //to be excluded in the next loop run
        int template_w= ceil(k_overlapping*templ.cols);
        int template_h= ceil(k_overlapping*templ.rows);
        int x=matchLoc.x-template_w/2;
        int y=matchLoc.y-template_h/2;

        //shrink template-mask size to avoid boundary violation
        if(y<0) y=0;
        if(x<0) x=0;
        //will template come beyond the mask?:if yes-cut off margin;
        if(template_w + x  > general_mask.cols)
            template_w= general_mask.cols-x;
        if(template_h + y  > general_mask.rows)
            template_h= general_mask.rows-y;

                               //set the negative mask to prevent repeating
        Mat template_mask=Mat::zeros(template_h,template_w, CV_8UC1);
        template_mask.copyTo(general_mask(cv::Rect(x, y, template_w, template_h)));

        /// Show me what you got on main image and on result (
        rectangle( img_display,matchLoc , Point( matchLoc.x + templ.cols ,    matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
        //small correction here-size of "result" is smaller
        rectangle( result,Point(matchLoc.x- templ.cols/2,matchLoc.y-     templ.rows/2) , Point( matchLoc.x + templ.cols/2 , matchLoc.y + templ.rows/2 ),     Scalar::all(0), 2, 8, 0 );
    }//for k= 0--5
}

void drawMatch(cv::Mat& templ, cv::Mat& frame, std::vector<cv::Point>& match_centers)
{
    for(size_t i=0; i < match_centers.size();i++)
    {
        rectangle(frame, cv::Point(match_centers[i].x, match_centers[i].y ), cv::Point(match_centers[i].x + templ.cols, match_centers[i].y + templ.rows), cv::Scalar(0,255,0), 2);
    }
}
