#include "sync_queue.h"
#include "template_matcher.h"

#include <QtDebug>
#include <QElapsedTimer>
#include <QThread>

#include "opencv2/opencv.hpp"

static const char* VIDEO_FILE_NAME = "field.mp4";
static const char* TEMPLATE_FILE_NAME = "template.png";
static const char* WINDOW = "Frame";

static void main_loop(cv::VideoCapture& cap, cv::Mat& templ, cv::Mat& mask, int thread_count);
static void updating_status(template_matcher& tm);

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE );

    // Lod the video and the template
    cv::VideoCapture cap(VIDEO_FILE_NAME);
    cv::Mat templ = cv::imread(TEMPLATE_FILE_NAME);
    cv::Mat full_templ = cv::imread(TEMPLATE_FILE_NAME, cv::IMREAD_UNCHANGED);

    if(templ.empty())
    {
        qCritical() << "Cannot load template" << TEMPLATE_FILE_NAME;
        return -1;
    }

    if(full_templ.empty())
    {
        qCritical() << "Cannot load full template " << TEMPLATE_FILE_NAME;
        return -1;
    }

    // Create the mask from the full_templ alpha channel (last channel)
    cv::Mat mask( full_templ.rows, full_templ.cols, CV_8UC1 );
    cv::extractChannel(full_templ, mask, 3);

    if(!cap.isOpened())
    {
        qCritical() << "Error opening video file" << VIDEO_FILE_NAME;
        return -1;
    }

    main_loop(cap, templ, mask, std::max(1, QThread::idealThreadCount() - 1));

    qDebug() << "Done.";

    cap.release();

    cv::destroyAllWindows();

    return 0;
}

/**
 * @brief Update the Frame window's status bar
 * @param tm The template_matcher fo get the status from
 */
void updating_status(template_matcher& tm)
{
    auto status = "Processing frame " +
            std::to_string(tm.get_frame_matched_counter()) +
            " of " + std::to_string(tm.get_frame_to_match_counter());
    cv::displayStatusBar(WINDOW, status);
}

/**
 * @brief Main processing loop
 *
 * Extract every frames of the video and process them
 * through the template_matcher::processFrame method.
 *
 * @param cap The VideoCapture to extract the frames from
 * @param templ The template to use for the match
 * @param mask The mask to use with the template
 * @param thread_count The number of consumers threads to use
 *          inside the template_matcher.
 */
void main_loop(cv::VideoCapture& cap, cv::Mat& templ, cv::Mat& mask, int thread_count)
{
    template_matcher tm(templ, mask, thread_count, WINDOW);

    // Current thread should prioritize running the processFrame method
    // This might not work on some OS
    QThread::currentThread()->setPriority(QThread::TimeCriticalPriority);

    while(1)
    {
        static int counter = 0;

        cv::Mat frame;

        // Read a video frame
        cap >> frame;

        // Are we at the end of the video file ?
        if(frame.empty())
        {
            break;
        }

        qDebug() << "Producing frame" << counter;
        counter += 1;

        updating_status(tm);
        tm.processFrame(frame);
    }

    // This might not work on some OS
    QThread::currentThread()->setPriority(QThread::NormalPriority);

    // Don't freeze the main UI
    // thread while killing consumers.
    // We detach here, but we won't loose
    // the detached thread as we are
    // waiting on tm below before exit.
    std::thread([&tm]()
    {
        tm.kill_consumers();
    }).detach();

    // To keep UI responsive while
    // consumers are still running
    while(tm.are_consumers_running())
    {
        updating_status(tm);
        cv::waitKey(100);
    }
}
