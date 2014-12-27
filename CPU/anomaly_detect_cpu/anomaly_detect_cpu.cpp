#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	double t = (double) getTickCount();

	const char* in = argv[1];
	const char* out = argv[2];
	if (in == NULL || out == NULL) {
		std::cout << "fail to pass parameters!" << endl;
		return 0;
	}
	std::cout << "video_in:" << in << "  video_out:" << out << std::endl;


	VideoCapture capture(in);

	if (!capture.isOpened()) {
		cout << "can not open video" << endl;
		return 0;
	}

	double fps = capture.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	int dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	int dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	int f_count = capture.get(CV_CAP_PROP_FRAME_COUNT); //get the height of frames of the video
	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;
	cout << "Frame count = " << f_count << endl;
    int count = 0;

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize,
			true); //initialize the VideoWriter objec

	Mat image;
	Mat prevImage;
	Mat diff;
	Mat grabbedImage;

	Mat frame;

	capture >> frame;
	cvtColor(frame, image, CV_RGB2GRAY);
	while (count < f_count-1) {
		grabbedImage = frame.clone();
		cv::GaussianBlur(grabbedImage, grabbedImage, Size(9, 9), 2, 2);
		prevImage = image.clone();
		cvtColor(grabbedImage, image, CV_RGB2GRAY);
		// perform ABS difference
		cv::absdiff(image, prevImage, diff);
		// do some threshold for wipe away useless details
		cv::threshold(diff, diff, 16, 255, CV_THRESH_BINARY);
		//sum of pixels
//		Scalar sca = sum(diff);
//		double x = sca.val[0]; //+sca.val[1]+sca.val[2];
		int x = countNonZero(diff);
		//cout << "x = " << x<< endl;
		if (x > 0) {
			// rectangle(frame, Point(10, 10),
			//          	   Point(0 + dWidth-10, 0 + dHeight-10),
			//          	   Scalar(0,255,255), 1, CV_AA, 0);
//			v_o.write(frame);
		}
		capture >> frame;
		count = count + 1;
	}
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}

