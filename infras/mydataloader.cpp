#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdbool>
#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "mydataloader.h"
#include "utils.h"

using namespace std;
using namespace cv;


TrafficTestLoader::TrafficTestLoader(string root, int batch_size, string dump_root, bool force_reload):
    n_images(12630), mbatch_size(batch_size), 
    image_arr(NULL), label_arr(NULL), image_shape({3, 96, 96}),
    batch_idx(0)
{
    const string image_dump_path = dump_root + "images.dmp";
    const string label_dump_path = dump_root + "labels.dmp";

    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    image_arr = new uchar[(n_images) * total_elems];
    label_arr = new int[n_images];

    if (force_reload or not check_file_existence(image_dump_path) or not check_file_existence(label_dump_path))
    {
        ifstream anno_in;
        char image_filename[10];        // store the image name
        int width, height, roi_x1, roi_y1, roi_x2, roi_y2;
        int classid;
        Mat original_img, resized_img;
        int target_height = image_shape[1], target_width = image_shape[2];
        
        string annotation_path = root + "GT-final_test.csv";
        anno_in.open(annotation_path, ios::in);

        // skip the first line
        anno_in.ignore(1000, NEWLINE);
        // set NULL as the last char of image_filename
        image_filename[9] = '\0';

        PRINT_DEBUG("Loading images ...\n");
        for (int image_cnt = 0; image_cnt < n_images; ++ image_cnt)
        {
            anno_in.read(image_filename, 9);
            anno_in.ignore(1);  // ignore delimiter
            string image_path = root + "GTSRB/Final_Test/Images/" + image_filename;

            // Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
            anno_in >> width; anno_in.ignore(1);
            anno_in >> height; anno_in.ignore(1);
            anno_in >> roi_x1; anno_in.ignore(1);
            anno_in >> roi_y1; anno_in.ignore(1);
            anno_in >> roi_x2; anno_in.ignore(1);
            anno_in >> roi_y2; anno_in.ignore(1);
            anno_in >> classid;
            anno_in.ignore(1000, NEWLINE);  // end this line

            // // test imag e path
            // cout << "reading " << image_path << "\t" << classid << endl;

            original_img = imread(image_path); // BGR image 
            resize(original_img, resized_img, Size(target_width, target_height), INTER_LINEAR);
            // imshow("Original image", original_img);
            // imshow("Resized image", resized_img);
            // waitKey(0);
            // // destroyWindow("window"); // does not work
            // destroyAllWindows();

            // just copy the resized image into the storage
            int idx = 0;
            for (int c = image_shape[0]-1; c >= 0; -- c)
            {
                for (int i = 0; i < image_shape[1]; ++ i)
                {
                    for (int j = 0; j < image_shape[2]; ++ j)
                    {
                        image_arr[image_cnt * total_elems + (idx ++)] = resized_img.at<Vec3b>(i, j)[c];
                    }
                }
            }
            label_arr[image_cnt] = classid;

            if (image_cnt % 1000 == 0)
            {
                if (DEBUG_FLAG) printf("[%d / %d] finished\n", image_cnt, n_images);
            }
        }

        anno_in.close();

        // dump the loaded data
        PRINT_DEBUG("Saving loaded data ... ");
        ofstream out_images(image_dump_path, ios::out | ios::binary);
        ofstream out_labels(label_dump_path, ios::out | ios::binary);

        out_images.write((const char *)image_arr, n_images * total_elems * sizeof(uchar) / sizeof(char));
        out_labels.write((const char *)label_arr, n_images * sizeof(int) / sizeof(char));

        out_images.close();
        out_labels.close();

        PRINT_DEBUG("Done!\n");
    }
    else
    {
        PRINT_DEBUG("Loading saved data ... ");
        ifstream in_images(image_dump_path, ios::in | ios::binary);
        ifstream in_labels(label_dump_path, ios::in | ios::binary);

        in_images.read((char *)image_arr, n_images * total_elems * sizeof(uchar) / sizeof(char));
        in_labels.read((char *)label_arr, n_images * sizeof(int) / sizeof(char));

        in_images.close();
        in_labels.close();

        PRINT_DEBUG("Done!\n");
    }

    // cout << (int)image_arr[10086] << "\t" << label_arr[10086] << endl; // should be: 139  7
}


TrafficTestLoader::~TrafficTestLoader()
{
    if (image_arr != NULL)
        delete [] image_arr;
    if (label_arr != NULL)
        delete [] label_arr;
}


bool TrafficTestLoader::check_file_existence(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


// Empty tensors are allowed as parameters. The shape will be assigned to the two tensors.
int TrafficTestLoader::next_batch(TypedTensor& inputs, Tensor<int>& labels)
{   // load batched inputs and labels to the provided arguments, return the number of samples
    // Since testing scans the images only once, so we can convert uchar pixels to floating numbers on-the-fly
    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    DTYPE *inputs_ptr = inputs.get_pointer();
    int *labels_ptr = labels.get_pointer();

    // allocate storage for inputs and labels if they have none
    if (inputs_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for inputs tensor\n");
        inputs_ptr = new DTYPE[mbatch_size * total_elems];
        inputs.set_pointer(inputs_ptr, true);
        inputs.set_shape({mbatch_size, image_shape[0], image_shape[1], image_shape[2]});
    }
    if (labels_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for labels tensor\n");
        labels_ptr = new int[mbatch_size];
        labels.set_pointer(labels_ptr, true);
        labels.set_shape({mbatch_size});
    }

    // compute no. of samples of this batch
    const int n_samples = min((batch_idx + 1) * mbatch_size, n_images) - batch_idx * mbatch_size;
    // cout << "n_samples: " << n_samples << endl;

    // return batch_idx-th batch, assume storage in `inputs` and `labels` are pre-allocated
    // CAUTION: NO SPACE CHECK! shape should be inputs: (B, C, H, W), labels: (B)
    // const vector<int> inputs_shape = inputs.get_shape();
    // const vector<int> labels_shape = labels.get_shape();
    // if (inputs_shape[0] < n_samples || inputs_shape[1] != image_shape[0] || 
    //     inputs_shape[2] != image_shape[1] || inputs_shape[3] != image_shape[2] || 
    //     labels_shape[0] < n_samples)
    // {
    //     throw invalid_argument("Incompatible shape!\n");   
    // }

    // convert uchar pixels to floating numbers on-the-fly, copy results to the tensor storage
    for (int i = 0; i < n_samples * total_elems; ++ i)
    {
        // Note: no scaling according to my pytorch implementation, please refer to poc/model/train.py
        inputs_ptr[i] = (DTYPE)image_arr[batch_idx * mbatch_size * total_elems + i]; // / 255.;
    }
    inputs.set_shape({n_samples, image_shape[0], image_shape[1], image_shape[2]});

    for (int i = 0; i < n_samples; ++ i)
    {
        labels_ptr[i] = label_arr[batch_idx * mbatch_size + i];
    }
    labels.set_shape({n_samples});

    batch_idx += 1;

    return n_samples;
}


// int main(void)
// {
//     TrafficTestLoader loader("../data/traffic/", 2, "./dump/", true);
//     // Mat img = imread("../data/traffic/GTSRB/Final_Test/Images/00000.ppm");
//     // printf("%d %d\n", img.size().height, img.size().width);
//     // printf("%d %d %d\n", img.at<Vec3b>(0, 0)[0], img.at<Vec3b>(0, 0)[1], img.at<Vec3b>(0, 0)[2]);
//     // printf("%d %d %d\n", img.at<Vec3b>(6, 6)[1], img.at<Vec3b>(23, 27)[0], img.at<Vec3b>(18, 50)[2]);

//     TypedTensor inputs;
//     Tensor<int> labels;

//     loader.next_batch(inputs, labels);
//     cout << "Inputs\n" << inputs << endl;
//     cout << "Labels\n" << labels << endl;

//     return 0;
// }