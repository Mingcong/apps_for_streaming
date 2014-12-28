#include "call_gpu.h"

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <cv.h>
#include <highgui.h>

using namespace caffe;
using namespace std;
using namespace cv;

//int main(int argc, char** argv) {
JNIEXPORT jint JNICALL Java_call_1gpu_app
  (JNIEnv *env, jclass, jint size, jstring mode, jstring in, jstring out) {


	double t = (double) getTickCount();
	Caffe::set_phase(Caffe::TEST);
	const int batch_size=size;
	const char *gpu;
	const char *input;
	const char *output;

	gpu = env->GetStringUTFChars(mode, 0);
	input = env->GetStringUTFChars(in, 0);
	output = env->GetStringUTFChars(out, 0);
	//Setting CPU or GPU
	if (strcmp(gpu, "GPU") == 0) {
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
	} else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	//get the net
	Net<float> caffe_test_net("/home/ideal/caffe-opencv/jni/proto.txt");
	//get trained net
	caffe_test_net.CopyTrainedLayersFrom(
			"/home/ideal/caffe-opencv/jni/bvlc_reference_caffenet.caffemodel");

	// Run ForwardPrefilled
	float loss;
//  const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled(&loss);

// Run AddImagesAndLabels and Forward

// cv::Mat image2 = cv::imread("p.png"); // or cat.jpg
//vector<cv::Mat> images(batch_size, image);
	vector<int> labels(batch_size, 0);
	const shared_ptr<ImageDataLayer<float> > image_data_layer =
			boost::static_pointer_cast < ImageDataLayer<float>
					> (caffe_test_net.layer_by_name("data"));
	double t1 = ((double) getTickCount() - t) / getTickFrequency();
	LOG(INFO) << " init time " << t1;

	double t2 = (double) getTickCount();
	int i = 0;
	vector < cv::Mat > images(batch_size);
	string image_name;
	string image_path;
	char img_list[255];
	char src[255];
	sprintf(src, "tar -xf /tmp/%s.tar -C /tmp", input);
	system(src);
	sprintf(img_list, "/tmp/%s/img_list.txt", input);

	ifstream infile(img_list, ios::in);
	while (getline(infile, image_name, '\n')) {
		image_path = "/tmp/" + image_name;
//		cout << image_path << endl;
		images[i] = imread(image_path);
		i = i + 1;
	}

//for(int k=0;k<1;k++){
	image_data_layer->AddImagesAndLabels(images, labels);

	double t3 = ((double) getTickCount() - t2) / getTickFrequency();
	LOG(INFO) << " load imgs time " << t3;

	double t4 = (double) getTickCount();
	vector<Blob<float>*> dummy_bottom_vec;
	const vector<Blob<float>*>& result = caffe_test_net.Forward(
			dummy_bottom_vec, &loss);

	//LOG(INFO)<< "Output result size: "<< result.size();
	// Now result will contain the argmax results.
	const float* argmaxs = result[1]->cpu_data();
	FILE *stream;
	char str[255];
	sprintf(str, "/tmp/%s", output);

	stream = fopen(str, "w+");

	LOG(INFO) << "Output result size: " << result.size();
	//for (int i = 0; i < 1; ++i) {
	for (int i = 0; i < result[1]->num(); ++i) {
//    LOG(INFO)<< " iteration: "<< k ;
		sprintf(str, "Image: %d    class: %.0f \n", i, argmaxs[i]);
//		printf("%s\n", str);
		fprintf(stream, str);
		LOG(INFO) << " Image: " << i << " class:" << argmaxs[i];
	}
	fclose(stream);
	double t5 = ((double) getTickCount() - t4) / getTickFrequency();
	LOG(INFO) << " loop time " << t5;
// }
	char del[255];
	sprintf(del, "rm -rf /tmp/%s", input);

	printf("%s\n", del);
	system(del);
	double t6 = ((double) getTickCount() - t) / getTickFrequency();
	LOG(INFO) << " total time " << t6;
	return 0;
}
