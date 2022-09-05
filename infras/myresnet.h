#ifndef MY_RESNET_H
#define MY_RESNET_H

#include <vector>

#include "common.h"

#define DTYPE_MIN -1e6
#define PADDING_VALUE 0.0

/** TO BE OPTIMIZED
 * change to inplace: BatchNorm2d, ReLU, operator+
 * improve efficiency: Conv2d
 **/


class InferenceBase
{ /* abstract base class for inference modules */
public:
    InferenceBase(): inplace(false) {}
    explicit InferenceBase(bool is_inplace): inplace(is_inplace) {}
    virtual ~InferenceBase() {}
    
    virtual TypedTensor forward(const TypedTensor& x) const =0;
    virtual void forward_inplace(TypedTensor& x) const { }
    virtual void load_from(ifstream &in_file) =0;
    
    virtual bool get_inplace(void) const { return inplace; }
    virtual void set_inplace(bool new_inplace) { inplace = new_inplace; }

protected:
    bool inplace;
};


void readline_from_ifstream(ifstream& in_file);


class Conv2d: public InferenceBase
{
public:
    Conv2d(
        int in_channels, int out_channels, vector<int> kernel_size, 
        int padding = 0, int stride = 1, bool bias = false);
    Conv2d(const Conv2d& obj);
    Conv2d(Conv2d && obj);
    // Conv2d(ifstream& in_file);
    ~Conv2d();
    TypedTensor forward(const TypedTensor& x) const;
    const vector<int> get_kernel_size(void) const;
    void load_weight_from(ifstream& in_file) { weight.load_from(in_file); }
    void load_weight_from(const char *filename) { weight.load_from(filename); }
    void load_from(ifstream& in_file);
    // void load_from(const char *filename);

private:
    TypedTensor weight;
    TypedTensor bias;
    int mpadding;
    int mstride;
};


TypedTensor unfoldTensor(const TypedTensor& x, const vector<int>& kernel_size, int padding, int stride);


class MaxPool2d: public InferenceBase
{
public:
    MaxPool2d(int kernel_size, int stride = -1, int padding = 0);
    MaxPool2d(const MaxPool2d& obj);
    MaxPool2d(MaxPool2d && obj);
    ~MaxPool2d();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
    int mkernel_size;
    int mstride;
    int mpadding;
};


class BatchNorm2d: public InferenceBase
{
public:
    BatchNorm2d(int num_features, double eps = 1e-5, bool is_inplace = true);
    BatchNorm2d(const BatchNorm2d& obj);
    BatchNorm2d(BatchNorm2d && obj);
    ~BatchNorm2d();
    TypedTensor forward(const TypedTensor& x) const;
    void forward_inplace(TypedTensor& x) const;
    void load_weight_from(ifstream& in_file) { weight.load_from(in_file); }
    void load_weight_from(const char *filename) { weight.load_from(filename); }
    void load_bias_from(ifstream& in_file) { bias.load_from(in_file); }
    void load_bias_from(const char *filename) { bias.load_from(filename); }
    void load_mean_from(ifstream& in_file) { running_mean.load_from(in_file); }
    void load_mean_from(const char *filename) { running_mean.load_from(filename); }
    void load_var_from(ifstream& in_file) { running_var.load_from(in_file); }
    void load_var_from(const char *filename) { running_var.load_from(filename); }
    void load_from(ifstream& in_file);

private:
    TypedTensor weight;
    TypedTensor bias;
    TypedTensor running_mean;
    TypedTensor running_var;
    TypedTensor processed_var;
    int mnum_features;
    double meps;
};


class Linear: public InferenceBase
{
public:
    Linear(int in_features, int out_features, bool bias = true);
    Linear(const Linear& obj);
    Linear(Linear && obj);
    ~Linear();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
    TypedTensor weight;
    TypedTensor mbias;
};


class ReLU: public InferenceBase
{
public:
    ReLU(bool is_inplace = true);
    ReLU(const ReLU& obj);
    ReLU(ReLU &&obj);
    ~ReLU();
    TypedTensor forward(const TypedTensor& x) const;
    void forward_inplace(TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:

};


class BasicBlock: public InferenceBase
{
public:
    BasicBlock(int in_channels, int out_channels);
    BasicBlock(const BasicBlock& obj);
    BasicBlock(BasicBlock && obj);
    ~BasicBlock();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu1;
    Conv2d conv2;
    BatchNorm2d bn2;
    Conv2d shortcut;
    ReLU relu2;
    bool use_shortcut;
};


class DuplicateBasicBlocks: public InferenceBase
{
public:
    DuplicateBasicBlocks(int n_repeat, int in_channels, int out_channels);
    ~DuplicateBasicBlocks();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
    vector<BasicBlock> basic_blocks;
};


class GlobalAveragePool2d: public InferenceBase
{
public:
    GlobalAveragePool2d();
    ~GlobalAveragePool2d();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
};


class Softmax: public InferenceBase
{
public:
    Softmax();
    ~Softmax();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
};


class ResNet34: public InferenceBase
{
public:
    ResNet34(int init_channel = 64);
    ~ResNet34();
    TypedTensor forward(const TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:
    vector<InferenceBase *> network;
};


#endif