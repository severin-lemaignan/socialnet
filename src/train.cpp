#include <opencv2/core.hpp>
#include <caffe/caffe.hpp>

#include "json.hpp"


using namespace std;
using namespace cv;
using namespace nlohmann; // json
using namespace caffe;

int main(int argc, char** argv)
{

    //::google::InitGoogleLogging(argv[0]);
    Caffe::Caffe::DeviceQuery();

    std::shared_ptr<caffe::Net<float>> net;

    net = make_shared<caffe::Net<float>>("share/models/socialnet.prototxt", caffe::TRAIN);
    //net->CopyTrainedLayersFrom("share/models/gaze.caffemodel");

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly 1 inputs.";
    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly 1 outputs.";

    auto input_layer = net->input_blobs()[0];

    auto num_channels_ = input_layer->channels();
    CHECK_EQ(num_channels_, 64) << "Input layer should have 64 channels.";



    float* input_data = input_layer->mutable_cpu_data();

    //for(size_t i = 0; i < features.size(); i++) {
    //    input_data[i] = features[i];
    //}


    net->Forward();

    /* Copy the output layer to a std::vector */
    auto output_layer = net->output_blobs()[0];
    auto output_data = output_layer->cpu_data();
    auto output = Point2f(output_data[0], output_data[1]);

    return 0;
}

