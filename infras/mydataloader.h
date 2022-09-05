#ifndef MY_DATA_LOADER_H
#define MY_DATA_LOADER_H

#include <cstdbool>
#include <string>
#include <vector>

#include "common.h"
#include "tensor.h"


class TrafficTestLoader
{
public:
    TrafficTestLoader(
        string root = "../data/traffic/",   // must have '/' as the last character
        int batch_size = 256, 
        string dump_root = "./dump/", 
        bool force_reload = false
    );
    ~TrafficTestLoader();
    int next_batch(TypedTensor& inputs, Tensor<int>& labels);
    void reset_batch(void) { batch_idx = 0; }
    int get_total_samples(void) { return n_images; }

private:
    const int n_images;         // total number of test images
    // string mroot;
    int mbatch_size;
    // string mdump_root;
    // bool mforce_reload;
    unsigned char *image_arr;           // 1-d storage for 4-d input with shape (N, C, H, W)
    int *label_arr;             // storage for labels
    const vector<int> image_shape;    // 3-d array, for (C, H, W) 
    int batch_idx;

    bool check_file_existence(const string& filename);
};


// class CIFAR10


#endif