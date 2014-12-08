//#include "com_smc_vidproc_call_gpu.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::gpu;





//JNIEXPORT jint JNICALL Java_com_smc_vidproc_call_1gpu_app
//  (JNIEnv *env, jclass, jstring video_in, jstring video_out)
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
	//string cascadeName = "/home/ideal/haarout.xml";
		string cascadeName = "/home/ideal/cars3.xml";

	VideoCapture capture_1(in_1);
	VideoCapture capture_2(in_2);
	VideoCapture capture_3(in_3);
	VideoCapture capture_4(in_4);

	if (!capture_1.isOpened() || !capture_2.isOpened()) {
		cout << "can not open video" << endl;
		return 0;
	}

//	env->ReleaseStringUTFChars(video_in, in);
//	env->ReleaseStringUTFChars(video_out, out);

	CascadeClassifier_GPU cascade_gpu;
	int gpuCnt = getCudaEnabledDeviceCount(); // gpuCnt >0 if CUDA device detected
	if (gpuCnt == 0) {
		cout << "can not gpu" << endl;
		return 0;
	}
	if (!cascade_gpu.load(cascadeName)) {
		return 0;
	}

	Mat frame, frame1, frame2, frame_1, frame_2, frame_3, frame_4;
    int count = 0;
	capture_1 >> frame_1;
	capture_2 >> frame_2;
	capture_3 >> frame_3;
	capture_4 >> frame_4;

	double fps = capture_1.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	int dWidth = capture_1.get(CV_CAP_PROP_FRAME_WIDTH)*2; //get the width of frames of the video
	int dHeight = capture_1.get(CV_CAP_PROP_FRAME_HEIGHT)*2; //get the height of frames of the video
	int f_count = capture_1.get(CV_CAP_PROP_FRAME_COUNT); //get the height of frames of the video

	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;
	cout << "Frame count = " << f_count << endl;

//	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
//
//	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec

	while (count < f_count-1) {
//		frame_roi = frame(Range(850,1080),Range(1100,1600));
//		combineTwoImages(frame,frame_1,frame_2);

        hconcat(frame_1,frame_2,frame1);
        hconcat(frame_3,frame_4,frame2);
        vconcat(frame1,frame2,frame);

		GpuMat cars;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		GpuMat gray_gpu(frame_gray);
		equalizeHist(frame_gray, frame_gray);

		int detect_num = cascade_gpu.detectMultiScale(gray_gpu, cars, 1.1, 2,
				Size(10, 10));
//		Mat obj_host;
//		cars.colRange(0, detect_num).download(obj_host);
//
//		Rect* ccars = obj_host.ptr<Rect>();
//
//		for (int i = 0; i < detect_num; ++i) {
//			Point pt1 = ccars[i].tl();
//			Size sz = ccars[i].size();
//			Point pt2(pt1.x + sz.width, pt1.y + sz.height);
//			rectangle(frame, pt1, pt2, Scalar(255));
//		}
////      imshow("cars", frame);
////		    if(waitKey(2)==27) break;
//
//		v_o.write(frame);

		capture_1 >> frame_1;
		capture_2 >> frame_2;
		capture_3 >> frame_3;
		capture_4 >> frame_4;

		count = count + 1;
//		cout << "count = " << count << endl;
	}

	capture_1.release();
	capture_2.release();
	capture_3.release();
	capture_4.release();
	// v_o.release();
	cascade_gpu.release();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}
