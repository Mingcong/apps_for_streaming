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

int main(int argc, char** argv) {

	double t = (double) getTickCount();
	if (argc < 5 || argc > 7) {
		LOG(ERROR) << "./test_net net_proto pretrained_net_proto batch_size "
				<< "[CPU/GPU] [Device ID]";
		return 1;
	}
	Caffe::set_phase(Caffe::TEST);

	int batch_size = 0;
	batch_size = atoi(argv[3]);
	//Setting CPU or GPU
	if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		if (argc == 6) {
			device_id = atoi(argv[5]);
		}
		Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
	} else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	//get the net
	Net<float> caffe_test_net(argv[1]);
	//get trained net
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);

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
	sprintf(src, "tar -xf /tmp/%s.tar -C /tmp", argv[6]);
	system(src);
	sprintf(img_list, "/tmp/%s/img_list.txt", argv[6]);

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
	//for (int i = 0; i < 1; ++i) {
	for (int i = 0; i < result[1]->num(); ++i) {
//    LOG(INFO)<< " iteration: "<< k ;
		LOG(INFO) << " Image: " << i << " class:" << argmaxs[i];
	}
	double t5 = ((double) getTickCount() - t4) / getTickFrequency();
	LOG(INFO) << " loop time " << t5;
// }
	char del[255];
	sprintf(del, "rm -rf /tmp/%s", argv[6]);

	printf("%s\n", del);
	system(del);
	double t6 = ((double) getTickCount() - t) / getTickFrequency();
	LOG(INFO) << " total time " << t6;
	return 0;
}