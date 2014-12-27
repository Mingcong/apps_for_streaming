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

	VideoCapture capture1(in);
	VideoCapture capture2(in);
	VideoCapture capture3(in);
	VideoCapture capture4(in);

	if (!capture1.isOpened()) {
		cout << "can not open video" << endl;
		return 0;
	}

	double fps = capture1.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	int dWidth = capture1.get(CV_CAP_PROP_FRAME_WIDTH)*2; //get the width of frames of the video
	int dHeight = capture1.get(CV_CAP_PROP_FRAME_HEIGHT)*2; //get the height of frames of the video
	int f_count = capture1.get(CV_CAP_PROP_FRAME_COUNT); //get the height of frames of the video
	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;
	cout << "Frame count = " << f_count << endl;
    int count = 0;


	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec

	GpuMat gpu_image;
	GpuMat diff;

	Mat frame, frame1, frame2, frame_1, frame_2, frame_3, frame_4;

	capture1 >> frame_1;
	capture2 >> frame_2;
	capture3 >> frame_3;
	capture4 >> frame_4;

    hconcat(frame_1,frame_2,frame1);
    hconcat(frame_3,frame_4,frame2);
    vconcat(frame1,frame2,frame);

	GpuMat frame_gpu(frame);
	GpuMat prevImage;
	gpu::cvtColor(frame_gpu, gpu_image, CV_RGB2GRAY);
	while (count < f_count-1) {
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
//		Scalar sca = sum(cpu_diff);
////		Scalar sca = gpu::sum(diff);
//		double x = sca.val[0]; // + sca.val[1] + sca.val[2];
		int x = countNonZero(cpu_diff);
//		cout << "x = " << x<< endl;
		if (x > 0) {
//			rectangle(frame, Point(10, 10),
//					Point(0 + dWidth - 10, 0 + dHeight - 10),
//					Scalar(0, 255, 255), 1, CV_AA, 0);
//			v_o.write(frame);
		}
//		imshow("anomaly", frame);
//		if (waitKey(2) == 27)
//			break;
		capture1 >> frame_1;
		capture2 >> frame_2;
		capture3 >> frame_3;
		capture4 >> frame_4;

	    hconcat(frame_1,frame_2,frame1);
	    hconcat(frame_3,frame_4,frame2);
	    vconcat(frame1,frame2,frame);
		count = count + 1;
	}
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;
}
