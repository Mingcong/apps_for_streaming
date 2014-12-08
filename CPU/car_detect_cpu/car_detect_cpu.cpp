//#include "com_smc_vidproc_call_cpu.h"

#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//JNIEXPORT jint JNICALL Java_com_smc_vidproc_call_1cpu_app(JNIEnv *env, jclass,
//		jstring video_in, jstring video_out) {
int main(int argc, char** argv) {
	double t = (double) getTickCount();
//	const char* in;
//	const char* out;
//	in = env->GetStringUTFChars(video_in, 0);
//	out = env->GetStringUTFChars(video_out, 0);
	const char* in = argv[1];
	const char* out = argv[2];
	if (in == NULL || out == NULL) {
		std::cout << "fail to pass parameters!" << endl;
		return 0;
	}
	std::cout << "video_in:" << in << "video_out:" << out << std::endl;

//	string cascadeName = "/home/ideal/haarout.xml";
	string cascadeName = "/home/ideal/cars3.xml";

	CascadeClassifier cascade;
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

//	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
//
//	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec


	//	env->ReleaseStringUTFChars(video_in, in);
//	env->ReleaseStringUTFChars(video_out, out);

	if (!cascade.load(cascadeName)) {
		cout << "can not find cascadeName" << endl;
		return 0;
	}

	Mat frame;

	capture >> frame;
	while (count < f_count-1) {

		std::vector<Rect> cars;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		cascade.detectMultiScale(frame_gray, cars, 1.05, 4, 0, Size(10, 10));

//		for (size_t i = 0; i < cars.size(); i++) {
//			Point pt1 = cars[i].tl();
//			Size sz = cars[i].size();
//			Point pt2(pt1.x + sz.width, pt1.y + sz.height);
//			rectangle(frame, pt1, pt2, Scalar(255));
//
//		}
//		v_o.write(frame);
		//imshow("cars", frame);
		//if(waitKey(2)==27) break;
		capture >> frame;
		count = count + 1;
	}
	capture.release();
	//v_o.release();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;

}
