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
//   const char* in;
//   const char* out;
//   in = env->GetStringUTFChars(video_in, 0);
//   out = env->GetStringUTFChars(video_out, 0);
	const char* in_1 = argv[1];
	const char* in_2 = argv[2];
	const char* in_3 = argv[3];
	const char* in_4 = argv[4];
	const char* out = argv[5];
	if (in_1 == NULL || in_2 == NULL || in_3 == NULL || in_4 == NULL ||out == NULL) {
		std::cout << "fail to pass parameters!" << endl;
		return 0; /* OutOfMemoryError already thrown */
	}
	std::cout << "video_in:" << in_1 << "  " << in_2 << "  video_out:" << out << std::endl;
	//	string cascadeName = "/home/ideal/haarout.xml";
	string cascadeName = "/home/ideal/cars3.xml";

	VideoCapture capture_1(in_1);
	VideoCapture capture_2(in_2);
	VideoCapture capture_3(in_3);
	VideoCapture capture_4(in_4);

	if (!capture_1.isOpened() || !capture_2.isOpened()) {
		cout << "can not open video" << endl;
		return 0;
	}

	double fps = capture_1.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	int dWidth = capture_1.get(CV_CAP_PROP_FRAME_WIDTH)*2; //get the width of frames of the video
	int dHeight = capture_1.get(CV_CAP_PROP_FRAME_HEIGHT)*2; //get the height of frames of the video
	int f_count = capture_1.get(CV_CAP_PROP_FRAME_COUNT); //get the height of frames of the video
	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;
	cout << "Frame count = " << f_count << endl;


	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec
//	env->ReleaseStringUTFChars(video_in, in);
//	env->ReleaseStringUTFChars(video_out, out);
	CascadeClassifier cascade;
	if (!cascade.load(cascadeName)) {
		cout << "can not find cascadeName" << endl;
		return 0;
	}
//	CascadeClassifier cascade1;
//	if (!cascade1.load(cascadeName1)) {
//		cout << "can not find cascadeName" << endl;
//		return 0;
//	}

	Mat frame, frame1, frame2, frame_1, frame_2, frame_3, frame_4;
    int count = 0;
	capture_1 >> frame_1;
	capture_2 >> frame_2;
	capture_3 >> frame_3;
	capture_4 >> frame_4;

	while (count < f_count-1) {
//		frame_roi = frame(Range(850,1080),Range(1100,1600));
        hconcat(frame_1,frame_2,frame1);
        hconcat(frame_3,frame_4,frame2);
        vconcat(frame1,frame2,frame);

		std::vector<Rect> cars;

		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		cascade.detectMultiScale(frame_gray, cars, 1.05, 4, 0, Size(10, 10));

//		for (size_t i = 0; i < cars.size(); i++) {
////			std::vector<Rect> cars1;
////			cascade1.detectMultiScale(frame_gray(cars[i]), cars1, 1.1, 2, 0, Size(30, 30));
////			for (size_t j = 0; j < cars1.size(); j++) {
//			Point pt1 = cars[i].tl();
//			Size sz = cars[i].size();
//			Point pt2(pt1.x + sz.width, pt1.y + sz.height);
//			rectangle(frame, pt1, pt2, Scalar(255));
////			}
//
//		}
//		v_o.write(frame);
//		imshow("cars", frame);
//		if(waitKey(2)==27) break;
		capture_1 >> frame_1;
		capture_2 >> frame_2;
		capture_3 >> frame_3;
		capture_4 >> frame_4;
		count = count + 1;
	}

	capture_1.release();
	capture_2.release();
	capture_3.release();
	capture_4.release();
	//v_o.release();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;

}
