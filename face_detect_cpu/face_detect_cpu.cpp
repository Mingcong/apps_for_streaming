#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
	std::cout << "video_in:" << in << "video_out:" << out << std::endl;

	string cascadeName =
			"/home/ideal/hadoop-1.2.1-cpu-gpu/classifier/haarcascade_frontalface_alt.xml";

	CascadeClassifier cascade;
	VideoCapture capture(in);
	if (!capture.isOpened()) {
		cout << "can not open input video" << endl;
		return 0;
	}

	double fps = capture.get(CV_CAP_PROP_FPS); //get the width of frames of the video
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << "FPS = " << fps << endl;

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	VideoWriter v_o(out, CV_FOURCC('D', 'I', 'V', 'X'), fps, frameSize, true); //initialize the VideoWriter objec

	if (!cascade.load(cascadeName)) {
		cout << "can not find cascadeName" << endl;
		return 0;
	}

	Mat frame;

	capture >> frame;
	while (frame.data) {

		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		cascade.detectMultiScale(frame_gray, faces, 1.2, 4, 0, Size(20, 20));

		for (size_t i = 0; i < faces.size(); i++) {
			Point pt1 = faces[i].tl();
			Size sz = faces[i].size();
			Point pt2(pt1.x + sz.width, pt1.y + sz.height);
			rectangle(frame, pt1, pt2, Scalar(255));

		}
		v_o.write(frame);
		//imshow("faces", frame);
		//if(waitKey(2)==27) break;
		capture >> frame;
	}
	capture.release();

	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	return 1;

}

