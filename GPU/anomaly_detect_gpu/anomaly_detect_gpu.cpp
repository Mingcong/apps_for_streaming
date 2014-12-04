#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
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
	std::cout << "video_in:" << in << "  video_out:" << out << std::endl;

	VideoCapture capture(in);

	if (!capture.isOpened()) {
		cout << "can not open video" << endl;
		return 0;
	}

	double fps = capture.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec

	GpuMat gpu_image;
	GpuMat diff;

	Mat frame;

	capture >> frame;

	GpuMat frame_gpu(frame);
	GpuMat prevImage;
	gpu::cvtColor(frame_gpu, gpu_image, CV_RGB2GRAY);
	while (frame.data) {
		frame_gpu.upload(frame);
		gpu::GaussianBlur(frame_gpu, frame_gpu, Size(9, 9), 2, 2);
		prevImage = gpu_image.clone();
		gpu::cvtColor(frame_gpu, gpu_image, CV_RGB2GRAY);
		// perform ABS difference
		gpu::absdiff(gpu_image, prevImage, diff);
		// do some threshold for wipe away useless details
		gpu::threshold(diff, diff, 16, 255, CV_THRESH_BINARY);
		//sum of pixels
		Mat cpu_diff(diff);
		Scalar sca = sum(cpu_diff);
//		Scalar sca = gpu::sum(diff);
		double x = sca.val[0]; // + sca.val[1] + sca.val[2];
//		cout << "x = " << x<< endl;
		if (x > 10) {
//			rectangle(frame, Point(10, 10),
//					Point(0 + dWidth - 10, 0 + dHeight - 10),
//					Scalar(0, 255, 255), 1, CV_AA, 0);
			v_o.write(frame);
		}
//		imshow("anomaly", frame);
//		if (waitKey(2) == 27)
//			break;
		capture >> frame;
	}
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}
