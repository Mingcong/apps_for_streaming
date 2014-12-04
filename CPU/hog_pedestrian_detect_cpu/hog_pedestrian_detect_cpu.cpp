#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	double t = (double) getTickCount();
//    setNumThreads(4);
	const char* in = argv[1];
	const char* out = argv[2];
	VideoCapture vc;
	Mat frame;
	if (in == NULL || out == NULL) {
		std::cout << "fail to pass parameters!" << endl;
		return 0;
	}

	// Create HOG descriptors and detectors here
	HOGDescriptor cpu_hog;
	cpu_hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//	namedWindow("people detector", 1);
	vc.open(in);
	if (!vc.isOpened())
		printf("could not open video\n");
	double fps = vc.get(CV_CAP_PROP_FPS);

	double dWidth = vc.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = vc.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	vc >> frame;

	VideoWriter video_writer(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize,
			true);
	Mat img_gray;

	while (!frame.empty()) {

		vector<Rect> found;
		cvtColor(frame, img_gray, CV_BGR2GRAY);

		// Perform HOG classification
		cpu_hog.detectMultiScale(img_gray, found, 0, Size(8, 8), Size(0, 0),
				1.05, 2);

		// Draw positive classified windows
		for (size_t i = 0; i < found.size(); i++) {
			Rect r = found[i];
			rectangle(frame, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
		}
		//   imshow("opencv_gpu_hog", img_to_show);
		//  waitKey(3);

		video_writer << frame;
		vc >> frame;

	}
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}
