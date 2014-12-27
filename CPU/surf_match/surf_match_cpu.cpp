#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void readme();

float innerAngle(float px1, float py1, float px2, float py2, float cx1,
		float cy1);
bool isSmallAngle(const std::vector<Point2f> scene_corners);

/** @function main */
int main(int argc, char** argv) {
	double t = (double) getTickCount();
	if (argc != 3) {
		readme();
		return -1;
	}

	Mat img_object = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	if (!img_object.data) {
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 500;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect(img_object, keypoints_object);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object);

//	VideoCapture capture(argv[2]);
//	if (!capture.isOpened()) {
//		cout << "can not open video" << endl;
//		return 0;
//	}
//
//	Mat frame, img_scene;
//	capture >> frame;
//	while (frame.data) {
//		cvtColor(frame, img_scene, CV_RGB2GRAY);
	int detect_count = 0;

	string image_name;
	ifstream infile(argv[2], ios::in);
	while (getline(infile, image_name, '\n')) {
		cout << line << endl;
		Mat img_scene = imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
		//imshow("11",img_scene);

//	    Mat img_scene = imread(argv[2] , CV_LOAD_IMAGE_GRAYSCALE );
		detector.detect(img_scene, keypoints_scene);
		extractor.compute(img_scene, keypoints_scene, descriptors_scene);
		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector < vector<DMatch> > matches;
		matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);
		std::vector < DMatch > good_matches;
		for (int k = 0; k < min(img_scene.rows - 1, (int) matches.size()); k++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
				{
			if ((matches[k][0].distance < 0.6 * (matches[k][4].distance))
					&& ((int) matches[k].size() <= 2
							&& (int) matches[k].size() > 0)) {
				good_matches.push_back(matches[k][0]);
			}
		}

		//printf("good_mathes : %d \n", good_matches.size() );

		Mat img_matches;
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		std::vector < Point2f > obj;
		std::vector < Point2f > scene;

		for (int i = 0; i < good_matches.size(); i++) {
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector < Point2f > obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_object.cols, 0);
		obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
		obj_corners[3] = cvPoint(0, img_object.rows);
		std::vector < Point2f > scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);
		bool falseDetect = isSmallAngle(scene_corners);
		if (!falseDetect) {
			cout
					<< "Detect!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
					<< endl;
			detect_count++;
		}
		else
			cout << "Not Detect!!!" << endl;

//		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0),
//				scene_corners[1] + Point2f(img_object.cols, 0),
//				Scalar(0, 255, 0), 4);
//		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0),
//				scene_corners[2] + Point2f(img_object.cols, 0),
//				Scalar(0, 255, 0), 4);
//		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0),
//				scene_corners[3] + Point2f(img_object.cols, 0),
//				Scalar(0, 255, 0), 4);
//		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0),
//				scene_corners[0] + Point2f(img_object.cols, 0),
//				Scalar(0, 255, 0), 4);
//
//		//-- Show detected matches
//		imshow("Good Matches & Object detection", img_matches);
//
//		waitKey(500);
//		capture >> frame;
	}
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "processing time: " << t << endl;
	cout << "detect_count: " << detect_count << endl;
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}

bool isSmallAngle(const std::vector<Point2f> scene_corners)

{
	const float minAngle = 70;
	const float maxAngle = 120;
	const float minHW = 15;

	float height = abs(scene_corners[3].y - scene_corners[0].y);
	float width = abs(scene_corners[1].x - scene_corners[0].x);
	// cout<< height<< " " << width << endl;
	if (height > minHW || width > minHW) {
		float left_top = innerAngle(scene_corners[1].x, scene_corners[1].y,
				scene_corners[3].x, scene_corners[3].y, scene_corners[0].x,
				scene_corners[0].y);
		float left_bot = innerAngle(scene_corners[0].x, scene_corners[0].y,
				scene_corners[2].x, scene_corners[2].y, scene_corners[3].x,
				scene_corners[3].y);
		float right_top = innerAngle(scene_corners[0].x, scene_corners[0].y,
				scene_corners[2].x, scene_corners[2].y, scene_corners[1].x,
				scene_corners[1].y);
		float right_bot = innerAngle(scene_corners[1].x, scene_corners[1].y,
				scene_corners[3].x, scene_corners[3].y, scene_corners[2].x,
				scene_corners[2].y);
		// std::cout << left_top << " " << right_top << " " << right_bot << " " << left_bot << std::endl;
		return left_bot < minAngle || left_top < minAngle
				|| right_top < minAngle || right_bot < minAngle
				|| left_bot > maxAngle || left_top > maxAngle
				|| right_top > maxAngle || right_bot > maxAngle;
	} else {
		return true;
	}
}

float innerAngle(float px1, float py1, float px2, float py2, float cx1,
		float cy1) {

	float angle;

	float ax = cx1 - px1;
	float ay = cy1 - py1;
	float bx = cx1 - px2;
	float by = cy1 - py2;
	float cost = (ax * bx + ay * by)
			/ sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
	if (cost != 0) {
		angle = acos(cost) * 180 / CV_PI;
	} else {
		angle = 90;
	}
	return angle;
}
