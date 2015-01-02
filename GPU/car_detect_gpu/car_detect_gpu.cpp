#include <iostream>
#include <fstream>
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

int main(int argc, char** argv) {
	double t = (double) getTickCount();

	const char* in = argv[1];
	const char* out = argv[2];
	if (in == NULL || out == NULL) {
		std::cout << "fail to pass parameters!" << endl;
		return 0;
	}
	std::cout << "video_in:" << in << "video_out:" << out << std::endl;
	string cascadeName = "/home/ideal/cars3.xml";

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
	char str[255];
    FILE *stream;
    stream = fopen(out, "w+");

//	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
//	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec

	CascadeClassifier_GPU cascade_gpu;
	int gpuCnt = getCudaEnabledDeviceCount(); // gpuCnt >0 if CUDA device detected
	if (gpuCnt == 0) {
		cout << "can not gpu" << endl;
		return 0;
	}
	if (!cascade_gpu.load(cascadeName)) {
		return 0;
	}

	Mat frame;
	capture >> frame;

	while (!frame.empty()) {
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
//		for (int i = 0; i < detect_num; ++i) {
//			Point pt1 = ccars[i].tl();
//			Size sz = ccars[i].size();
//			Point pt2(pt1.x + sz.width, pt1.y + sz.height);
//			rectangle(frame, pt1, pt2, Scalar(255));
//		}
//		imshow("cars", frame);
//		if (waitKey(2) == 27)
//			break;
//		v_o.write(frame);
		count = count + 1;

		if(detect_num > 0) {
//			cout << "frame: " << count << " cars = " << detect_num << endl;
			sprintf(str, "frame: %d   cars: %d\n", count, detect_num);
			fprintf(stream, str);
		}

		capture >> frame;
	}

	capture.release();
	fclose(stream);
//	v_o.~VideoWriter();
	cascade_gpu.release();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}
