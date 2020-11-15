#include "template_matching.h"
#include "sync_queue.h"

#include <QApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QThread>

#include "opencv2/opencv.hpp"

#include <filesystem>
#include <mutex>
#include <thread>

void fastLoopCode();
void viewLoopCode();

struct Frame
{
    uint id;
    cv::Mat* mat;
};

std::mutex mtx;
sync_queue<Frame> sq;

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::thread fastLoop([]()
    {
        fastLoopCode();
    });

    fastLoop.join();

    std::thread viewLoop([]()
    {
        viewLoopCode();
    });

    viewLoop.join();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}

void enqueue_mat(cv::Mat& mat)
{
    static uint frame_count = 0;

    // Will be deleted by consumer
    cv::Mat* copy = new cv::Mat(mat.size(), mat.type());
    mat.copyTo(*copy);

    sq.push({frame_count, copy});
    frame_count += 1;
}

void fastLoopCode()
{
    cv::VideoCapture cap("/home/vm/imagia/field.mp4");

    // Check if camera opened successfully
    if(!cap.isOpened())
    {
        qDebug() << "Error opening video stream or file" << endl;
        return;
    }

    while(true)
    {
        static uint counter = 0;

        // Capture frame-by-frame
        cv::Mat mat;
        cap >> mat;

        enqueue_mat(mat);

        if(mat.empty())
        {
            break;
        }

        qDebug() << "Produce" << counter;
        counter += 1;
    }

    cap.release();
}

void viewLoopCode()
{
    namedWindow("frame", cv::WINDOW_AUTOSIZE);
    cv::waitKey(1);

    while(1)
    {
        static uint counter = 0;

        // Retrieve frame
        Frame f;
        if(sq.pop(f))
        {
            QElapsedTimer timer;
            timer.start();

            if(f.mat->empty())
            {
                delete f.mat;
                break;
            }

            qDebug() << "Showing" << f.id;
            cv::imshow("frame", *f.mat);

            qDebug() << "Consume" << counter;
            counter += 1;

            delete f.mat;

            cv::waitKey(1);

            qDebug() << "elapsed1=" << timer.elapsed() << "ms";
        }
        else
        {
            qDebug() << "Waiting";
            cv::waitKey(1);
        }
    }

    cv::waitKey(1);
}

int main2(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    qDebug() << "CWD:" << std::filesystem::current_path().c_str();
    qDebug() << cv::getBuildInformation().c_str();

//    QApplication a(argc, argv);
//    TemplateMatching w;
//    w.show();
//    return a.exec();

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture cap("/home/vm/imagia/field.mp4");

    // Check if camera opened successfully
    if(!cap.isOpened())
    {
        qDebug() << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1)
    {
//        const int sleep_period_ms = 17;
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

//        ulong sleep_ms = sleep_period_ms - timer.elapsed();
//        qDebug() << "sleeping" << sleep_ms << "ms";
//        QThread::msleep(sleep_ms);

        qDebug() << "elapsed1=" << timer.elapsed() << "ms";



        // Display the resulting frame
        imshow("Frame", frame );

        qDebug() << "elapsed2=" << timer.elapsed() << "ms";

//        // Press  ESC on keyboard to exit
        char c=(char)cv::waitKey(1);
        if(c==27)
          break;

        qDebug() << "elapsed3=" << timer.elapsed() << "ms";
        qDebug() << "==============================";

//        qDebug() << timer.elapsed();
//        qDebug() << sleep_period_ms - timer.elapsed();


//        ulong sleep_ms = sleep_period_ms - timer.elapsed();
//        qDebug() << "sleeping" << sleep_ms << "ms";
//        QThread::msleep(sleep_ms);
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
