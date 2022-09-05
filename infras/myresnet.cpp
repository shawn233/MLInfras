#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdbool>
#include <vector>
#include <string>
#include <stdexcept>

#include "myresnet.h"
#include "utils.h"
extern "C" {
    #include "./openblas/include/cblas.h"
} 

using namespace std;


Conv2d::Conv2d(
    int in_channels, 
    int out_channels, 
    vector<int> kernel_size, 
    int padding, 
    int stride,
    bool bias
):
    // weight init: four-dim tensor
    // - tensor size is (C_{out} x C_{in} x K[0] x K[1])
    // - if `kernel_size` uses one value to represent a 2-dim kernel size, then K[1] = K[0]
    weight(
        4, out_channels, in_channels, kernel_size[0], 
        (kernel_size.size() > 1)?kernel_size[1]:kernel_size[0]),
    // bias init: uses no bias by default (bias is currently not supported)
    bias(),
    // by default, padding = 0, and stride = 1
    mpadding(padding), mstride(stride)
{
    PRINT_CONSTRUCTOR("Conv2d constructor\n");
    if (bias)
    {
        throw invalid_argument("conv bias is not implemented yet!");
    }
}


Conv2d::Conv2d(const Conv2d& obj):
    weight(obj.weight), bias(obj.bias), 
    mpadding(obj.mpadding), mstride(obj.mstride)
{
    PRINT_CONSTRUCTOR("Conv2d copy constructor\n");
}


Conv2d::Conv2d(Conv2d && obj):
    weight(move(obj.weight)), bias(move(obj.bias)),
    mpadding(move(obj.mpadding)), mstride(move(obj.mstride))
{
    PRINT_CONSTRUCTOR("Conv2d move constructor\n");
}


Conv2d::~Conv2d()
{

}


const vector<int> Conv2d::get_kernel_size(void) const
{
    const vector<int>& weight_shape = weight.get_shape();
    vector<int> ret({weight_shape[2], weight_shape[3]});
    return ret;
}



TypedTensor Conv2d::forward(const TypedTensor& x) const
{   // x: (N, C, H, W)
    MyTimer my_timer;

    // unfold the input tensor
    TypedTensor unf_x = unfoldTensor(x, this->get_kernel_size(), mpadding, mstride);
    PRINT_TIMING("[conv2d] unfolding", my_timer.elapsed_time(true));

    const vector<int>& x_shape = x.get_shape();
    const vector<int>& w_shape = weight.get_shape();
    const vector<int>& unf_x_shape = unf_x.get_shape();
    
    int batch_size = x_shape[0];
    int c_in = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];
    int c_out = w_shape[0];
    int ks0 = w_shape[2];
    int ks1 = w_shape[3];

    int h_out = (int)floor((double)(h_in + 2 * mpadding - ks0) / mstride + 1);
    int w_out = (int)floor((double)(w_in + 2 * mpadding - ks1) / mstride + 1);

    // flatten the last three dimensions of the kernel -> (C_out, K = C_in*ks0*ks1)
    vector<int> flat_shape({c_out, c_in * ks0 * ks1});
    TypedTensor flat_w(weight.get_pointer(), flat_shape); // share memory with weight
    // printf("Flattern weight shape: ");
    // print_vector(flat_w.get_shape());

    PRINT_TIMING("[conv2d] preparation", my_timer.elapsed_time(true));

    // batched matrix multiply: unf_x (N, K, H_out*W_out) x flat_w (C_out, K)
    TypedTensor ret(3, batch_size, c_out, h_out * w_out);

    PRINT_TIMING("[conv2d] return allocation", my_timer.elapsed_time(true));

    // C := A (M, K) * B (K, N) + C (M, N)
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA, transB;
    blasint M, N, K;
    blasint lda, ldb, ldc;  // when order is RowMajor, these represent no. of columns

    order = CblasRowMajor;
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    M = c_out;
    N = h_out * w_out;
    K = flat_shape[1];
    lda = K;
    ldb = N;
    ldc = N;

    DTYPE *A = flat_w.get_pointer();
    DTYPE *B = unf_x.get_pointer();
    DTYPE *C = ret.get_pointer();

    for (int i = 0; i < unf_x_shape[0]; ++ i)
    {   // ret[i] = flat_w * unf_x[i]
        cblas_dgemm(order, transA, transB, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
        B += (unf_x_shape[1] * unf_x_shape[2]); // unf_x[i]
        C += (flat_shape[0] * unf_x_shape[2]);  // ret[i] 
    }

    // `ret` shape: (N, C_out, H_out*W_out)
    // reshape to (N, C_out, H_out, W_out)
    ret.set_shape({batch_size, c_out, h_out, w_out});

    PRINT_TIMING("[conv2d] calculation", my_timer.elapsed_time(true));

    return ret;
}


/**
 * Unfold an input 4-d tensor (batched feature maps) into a 3-d tensor 
 * by scanning with a kernel.
 * 
 * x: (N, C, H_{in}, W_{in})
 * H_{out} = floor((H_{in} + 2 x padding - kernel_size[0]) / stride + 1)
 * W_{out} = floor((W_{in} + 2 x padding - kernel_size[1]) / stride + 1)
 * 
 * Output tensor: (N, C * kernel_size[0] * kernel_size[1], H_{out} * W_{out})
 **/
TypedTensor unfoldTensor(const TypedTensor& x, const vector<int>& kernel_size, int padding, int stride)
{
    if (kernel_size.size() < 1)
    {
        throw invalid_argument("No kernel size");
    }
    else if (padding < 0)
    {
        throw invalid_argument("Invalid padding");
    }
    else if (stride < 1)
    {
        throw invalid_argument("Invalid stride");
    }

    MyTimer my_timer;
    
    int ks0, ks1;
    ks0 = ks1 = kernel_size[0];
    if (kernel_size.size() > 1)
    {
        ks1 = kernel_size[1];
    }

    const vector<int> x_shape = x.get_shape();
    if (x_shape.size() != 4)
    {
        throw invalid_argument("Input tensor is not 4-d");
    }

    int batch_size = x_shape[0];
    int n_channels = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];

    int h_out = (int)floor((double)(h_in + 2 * padding - kernel_size[0]) / stride + 1);
    int w_out = (int)floor((double)(w_in + 2 * padding - kernel_size[1]) / stride + 1);

    const int ret_dim2 = n_channels * ks0 * ks1;
    const int ret_dim3 = h_out * w_out;
    TypedTensor ret(3, batch_size, ret_dim2, ret_dim3);

    int n2, n3; // counters for indexing the `ret` tensor 

    PRINT_TIMING("[conv2d.unfold] preparation", my_timer.elapsed_time(true));

    DTYPE *ret_ptr = ret.get_pointer();
    DTYPE *x_ptr = x.get_pointer();

    for (int sample_idx = 0; sample_idx < batch_size; ++ sample_idx)
    {   // for each sample, we scan by a window sized the same as the kernel
        n3 = 0;
        for (int i = -padding; i + ks0 <= h_in + padding; i += stride)
        {
            for (int j = -padding; j + ks1 <= w_in + padding; j += stride)
            {
                // (i, j) is the starting position of a kernel window (upper left corner)
                n2 = 0;
                for (int c = 0; c < n_channels; ++ c)
                {   // scan by channel
                    for (int ki = 0; ki < ks0; ++ ki)
                    {   
                        for (int kj = 0; kj < ks1; ++ kj)
                        {
                            if (i + ki < 0 || i + ki >= h_in || j + kj < 0 || j + kj >= w_in)
                            {   // the value depends on the padding scheme, default to zero padding
                                // ret.at(sample_idx, n2, n3) = PADDING_VALUE;
                                ret_ptr[((sample_idx * ret_dim2) + n2) * ret_dim3 + n3] = PADDING_VALUE;
                                // ret.at_3d(sample_idx, n2, n3) = PADDING_VALUE;
                            }
                            else
                            {
                                // ret.at(sample_idx, n2, n3) = x.at(sample_idx, c, i + ki, j + kj);
                                ret_ptr[((sample_idx * ret_dim2) + n2) * ret_dim3 + n3] = 
                                    x_ptr[((((sample_idx * n_channels) + c) * h_in) + (i + ki)) * w_in + (j + kj)];
                                // ret.at_3d(sample_idx, n2, n3) = x.at_4d(sample_idx, c, i + ki, j + kj);
                            }
                            ++ n2; // we move to the next element
                        }
                    }
                }
                ++ n3; // the next starting point 
            }
        }
    }

    PRINT_TIMING("[conv2d.unfold] unfolding", my_timer.elapsed_time(true));

    return ret;
}


void Conv2d::load_from(ifstream& in_file)
{
    weight.load_from(in_file);
    if (bias.get_shape().size() > 0)
    {
        bias.load_from(in_file);
    }
}


void readline_from_ifstream(ifstream& in_file)
{
    char ch;
    do
    {
        in_file >> ch;
    } while (ch != NEWLINE);
}


MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding):
    mkernel_size(kernel_size), mstride(stride), mpadding(padding)
{   
    PRINT_CONSTRUCTOR("MaxPool2d constructor\n");
    if (kernel_size <= 0)
    {
        throw invalid_argument("Invalid kernel size");
    }
    if (stride < -1)
    {
        throw invalid_argument("Invalid stride");
    }
    if (padding < 0)
    {
        throw invalid_argument("Invalid padding");
    }
    // stride defaults to kernel_size 
    if (mstride == -1)
    {
        mstride = kernel_size;
    }
}


MaxPool2d::MaxPool2d(const MaxPool2d& obj):
    mkernel_size(obj.mkernel_size),
    mstride(obj.mstride),
    mpadding(obj.mpadding)
{
    PRINT_CONSTRUCTOR("MaxPool2d copy constructor\n");
}


MaxPool2d::MaxPool2d(MaxPool2d && obj):
    mkernel_size(move(obj.mkernel_size)),
    mstride(move(obj.mstride)),
    mpadding(move(obj.mpadding))
{
    PRINT_CONSTRUCTOR("MaxPool2d move constructor\n");
}


MaxPool2d::~MaxPool2d()
{

}


TypedTensor MaxPool2d::forward(const TypedTensor& x) const
{
    const vector<int>& x_shape = x.get_shape();

    int batch_size = x_shape[0];
    int c_in = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];

    int h_out = (int)floor((double)(h_in + 2 * mpadding - mkernel_size) / mstride + 1);
    int w_out = (int)floor((double)(w_in + 2 * mpadding - mkernel_size) / mstride + 1);

    TypedTensor ret(4, batch_size, c_in, h_out, w_out);
    DTYPE *ret_ptr = ret.get_pointer();
    DTYPE *x_ptr = x.get_pointer();

    int idx = 0; // for ret_ptr
    for (int n_sample = 0; n_sample < batch_size; ++ n_sample)
    {
        for (int n_channel = 0; n_channel < c_in; ++ n_channel)
        {
            for (int i = -mpadding; i + mkernel_size <= h_in + mpadding; i += mstride)
            {
                for (int j = -mpadding; j + mkernel_size <= w_in + mpadding; j += mstride)
                {
                    ret_ptr[idx] = DTYPE_MIN;
                    for (int ki = 0; ki < mkernel_size; ++ ki)
                    {
                        for (int kj = 0; kj < mkernel_size; ++ kj)
                        {
                            if (i + ki < 0 || i + ki >= h_in || j + kj < 0 || j + kj >= w_in)
                            {
                                ret_ptr[idx] = max(ret_ptr[idx], PADDING_VALUE);
                            }
                            else
                            {
                                ret_ptr[idx] = max(
                                    ret_ptr[idx], 
                                    x_ptr[((((n_sample * c_in) + n_channel) * h_in + (i + ki)) * w_in) + (j + kj)]);
                                    // x.at(n_sample, n_channel, i + ki, j + kj));
                            }
                        }
                    }
                    ++ idx;
                }
            }
        }
    }

    return ret;

    // TypedTensor unf_x = unfoldTensor(x, {mkernel_size, mkernel_size}, mpadding, mstride);

    // const vector<int>& x_shape = x.get_shape();

    // int batch_size = x_shape[0];
    // int c_in = x_shape[1];
    // int h_in = x_shape[2];
    // int w_in = x_shape[3];

    // int h_out = (int)floor((double)(h_in + 2 * mpadding - mkernel_size) / mstride + 1);
    // int w_out = (int)floor((double)(w_in + 2 * mpadding - mkernel_size) / mstride + 1);

    // TypedTensor ret(batch_size, c_in, h_out * w_out);

    // for (int i = 0; i < batch_size; ++ i)
    // {

    // }

    // ret.local_reshape(4, batch_size, c_in, h_out, w_out);

    // return ret;
}


void MaxPool2d::load_from(ifstream& in_file)
{
    // no parameter to load
}


/** PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
 * "Also by default, during training this layer keeps running estimates of 
 * its computed mean and variance, which are then used for normalization 
 * during evaluation."
 **/
BatchNorm2d::BatchNorm2d(int num_features, double eps, bool is_inplace):
    InferenceBase(is_inplace),
    weight(1, num_features), bias(1, num_features), 
    running_mean(1, num_features), running_var(1, num_features), 
    processed_var(1, num_features),
    mnum_features(num_features), meps(eps)
{
    PRINT_CONSTRUCTOR("BatchNorm2d constructor\n");
}


BatchNorm2d::BatchNorm2d(const BatchNorm2d& obj):
    InferenceBase(obj.inplace),
    weight(obj.weight), bias(obj.bias),
    running_mean(obj.running_mean), running_var(obj.running_var),
    processed_var(obj.processed_var),
    mnum_features(obj.mnum_features), meps(obj.meps)
{
    PRINT_CONSTRUCTOR("BatchNorm2d copy constructor\n");
}


BatchNorm2d::BatchNorm2d(BatchNorm2d && obj):
    InferenceBase(obj.inplace),
    weight(move(obj.weight)), bias(move(obj.bias)),
    running_mean(move(obj.running_mean)), 
    running_var(move(obj.running_var)),
    processed_var(move(obj.processed_var)),
    mnum_features(move(obj.mnum_features)),
    meps(move(obj.meps))
{
    PRINT_CONSTRUCTOR("BatchNorm2d move constructor\n");
}



BatchNorm2d::~BatchNorm2d()
{

}


TypedTensor BatchNorm2d::forward(const TypedTensor& x) const
{   // y = ( (x - mean) / sqrt(var + eps) ) * weight + bias
    // running_mean, running_var, weight, and bias, all of which have size (n_channels)
    // x: (N, C, H, W)
    TypedTensor ret(x);

    const vector<int>& x_shape = x.get_shape();
    int batch_size = x_shape[0];
    int n_channels = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];

    for (int n_sample = 0; n_sample < batch_size; ++ n_sample)
    {
        for (int c = 0; c < n_channels; ++ c)
        {
            for (int i = 0; i < h_in; ++ i)
            {
                for (int j = 0; j < w_in; ++ j)
                {
                    ret.at(n_sample, c, i, j) = 
                        (ret.at(n_sample, c, i, j) - running_mean.at(c)) /
                        (sqrt(running_var.at(c) + meps)) * weight.at(c) +
                        bias.at(c);
                }
            }
        }
    }

    return ret;
}


void BatchNorm2d::forward_inplace(TypedTensor& x) const
{
    const vector<int>& x_shape = x.get_shape();
    int batch_size = x_shape[0];
    int n_channels = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];

    DTYPE *w_ptr = weight.get_pointer();
    DTYPE *b_ptr = bias.get_pointer();
    DTYPE *rm_ptr = running_mean.get_pointer();
    DTYPE *pv_ptr = processed_var.get_pointer();

    DTYPE *x_ptr = x.get_pointer();
    int pos;

    // PRINT_DEBUG("> inplace batchnorm2d\n");

    for (int n_sample = 0; n_sample < batch_size; ++ n_sample)
    {
        for (int c = 0; c < n_channels; ++ c)
        {
            for (int i = 0; i < h_in; ++ i)
            {
                for (int j = 0; j < w_in; ++ j)
                {
                    pos = ((((n_sample * n_channels) + c) * h_in) + i) * w_in + j;
                    x_ptr[pos] = 
                        (x_ptr[pos] - rm_ptr[c]) /
                        pv_ptr[c] * w_ptr[c] +
                        b_ptr[c];
                }
            }
        }
    }
}


void BatchNorm2d::load_from(ifstream& in_file)
{
    // follow the order of pytorch
    weight.load_from(in_file);          // weight
    bias.load_from(in_file);            // bias
    running_mean.load_from(in_file);    // running mean
    running_var.load_from(in_file);     // running var
    // num_batches_tracker

    DTYPE *pv_ptr = processed_var.get_pointer();
    DTYPE *rv_ptr = running_var.get_pointer();

    // compute processed_var from running_var
    for (int i = 0; i < mnum_features; ++ i)
    {
        pv_ptr[i] = sqrt(rv_ptr[i] + meps);
    }
}


// y = xA^T + b (reference: pytorch doc)
Linear::Linear(int in_features, int out_features, bool bias):
    weight(2, out_features, in_features) 
{
    PRINT_CONSTRUCTOR("Linear constructor\n");
    if (bias)
    {
        mbias = TypedTensor(1, out_features); // assignment operator
    }
}


Linear::Linear(const Linear& obj):
    weight(obj.weight), mbias(obj.mbias)
{
    PRINT_CONSTRUCTOR("Linear copy constructor\n");
}


Linear::Linear(Linear && obj):
    weight(move(obj.weight)),
    mbias(move(obj.mbias))
{
    PRINT_CONSTRUCTOR("Linear move constructor\n");
}


Linear::~Linear()
{

}


TypedTensor Linear::forward(const TypedTensor& x) const
{   // x: (N, in_feats)
    const vector<int>& x_shape = x.get_shape();
    int batch_size = x_shape[0];
    const vector<int>& w_shape = weight.get_shape();
    int in_feats = w_shape[1];
    int out_feats = w_shape[0];

    TypedTensor ret(2, batch_size, out_feats);

    // C := A (M, K) * TRANS( B (N, K) ) + C (M, N)
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA, transB;
    blasint M, N, K;
    blasint lda, ldb, ldc;  // when order is RowMajor, these represent no. of columns

    order = CblasRowMajor;
    transA = CblasNoTrans;
    transB = CblasTrans;
    M = batch_size;
    N = out_feats;
    K = in_feats;
    lda = K;
    ldb = K; // since B is transposed, ldb = K instead of N 
    ldc = N;

    DTYPE *A = x.get_pointer();
    DTYPE *B = weight.get_pointer();
    DTYPE *C = ret.get_pointer(); // broadcast mbias to `ret`
    DTYPE *mbias_ptr = mbias.get_pointer();
    // PRINT_DEBUG("Broadcasting bias ...\n");
    // if (DEBUG_FLAG)
    // {
    //     printf("x.shape: ");
    //     print_vector(x_shape);
    //     printf("weight.shape: ");
    //     print_vector(w_shape);
    //     printf("ret.shape: ");
    //     print_vector(ret.get_shape());
    // }
    for (int i = 0; i < batch_size; ++ i)
    {
        for (int j = 0; j < out_feats; ++ j)
        {
            C[i * out_feats + j] = mbias_ptr[j];
        }
    }

    cblas_dgemm(order, transA, transB, M, N, K, 1.0, A, lda, B, ldb, 1.0, C, ldc);

    return ret;
}


void Linear::load_from(ifstream& in_file)
{
    weight.load_from(in_file);
    if (mbias.get_shape().size() > 0)
    {
        mbias.load_from(in_file);
    }
}


ReLU::ReLU(bool is_inplace):
    InferenceBase(is_inplace)
{
    PRINT_CONSTRUCTOR("ReLU constructor\n");
}


ReLU::ReLU(const ReLU& obj):
    InferenceBase(obj.inplace)
{
    PRINT_CONSTRUCTOR("ReLU copy constructor\n");
}


ReLU::ReLU(ReLU &&obj):
    InferenceBase(obj.inplace)
{
    PRINT_CONSTRUCTOR("ReLU move constructor\n");
}


ReLU::~ReLU()
{
    // PRINT_DEBUG("ReLU destructor\n");
}


TypedTensor ReLU::forward(const TypedTensor& x) const
{   // x: any shape
    TypedTensor ret(x);

    DTYPE *ret_ptr = ret.get_pointer();
    for (int i = 0; i < ret.get_total_elems(); ++ i)
    {
        if (ret_ptr[i] < 0.0)
            ret_ptr[i] = 0.0;
        // ret_ptr[i] = (ret_ptr[i] < 0.0)? 0.0: ret_ptr[i];
    }

    return ret;
}


void ReLU::forward_inplace(TypedTensor& x) const
{
    DTYPE* x_ptr = x.get_pointer();
    const vector<int> x_shape = x.get_shape();
    const int total_elems = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3];

    // PRINT_DEBUG("> inplace relu\n");

    for (int i = 0; i < total_elems; ++ i)
    {
        if (x_ptr[i] < 0.0)
            x_ptr[i] = 0.0;
        // x_ptr[i] = max(x_ptr[i], 0.0);
    }
}


void ReLU::load_from(ifstream& in_file)
{
    // no parameter to load
}


BasicBlock::BasicBlock(int in_channels, int out_channels):
    conv1(in_channels, out_channels, {3, 3}, 1, (in_channels == out_channels)? 1: 2),
    bn1(out_channels),
    relu1(),
    conv2(out_channels, out_channels, {3, 3}, 1, 1),
    bn2(out_channels),
    shortcut(in_channels, out_channels, {1, 1}, 0, 2),
    relu2(),
    use_shortcut(in_channels != out_channels)
{
    PRINT_CONSTRUCTOR("BasicBlock constructor\n");
}


BasicBlock::BasicBlock(const BasicBlock& obj):
    conv1(obj.conv1), bn1(obj.bn1), relu1(obj.relu1),
    conv2(obj.conv2), bn2(obj.bn2), shortcut(obj.shortcut),
    relu2(obj.relu2), use_shortcut(obj.use_shortcut)
{
    PRINT_CONSTRUCTOR("BasicBlock copy constructor\n");
}


BasicBlock::BasicBlock(BasicBlock && obj):
    conv1(move(obj.conv1)), bn1(move(obj.bn1)), relu1(move(obj.relu1)),
    conv2(move(obj.conv2)), bn2(move(obj.bn2)), shortcut(move(obj.shortcut)),
    relu2(move(obj.relu2)), use_shortcut(move(obj.use_shortcut))
{
    PRINT_CONSTRUCTOR("BasicBlock move constructor\n");
}


BasicBlock::~BasicBlock()
{

}


TypedTensor BasicBlock::forward(const TypedTensor& x) const
{
    MyTimer my_timer;
    
    TypedTensor ret = conv1.forward(x);
    PRINT_TIMING("conv1", my_timer.elapsed_time(true));
    
    // ret = bn1.forward(ret);
    bn1.forward_inplace(ret);
    PRINT_TIMING("bn1", my_timer.elapsed_time(true));
    
    relu1.forward_inplace(ret);
    PRINT_TIMING("relu1", my_timer.elapsed_time(true));

    ret = conv2.forward(ret);
    PRINT_TIMING("conv2", my_timer.elapsed_time(true));

    bn2.forward_inplace(ret);
    PRINT_TIMING("bn2", my_timer.elapsed_time(true));

    if (use_shortcut)
    {
        // PRINT_DEBUG("use shortcut\n");
        ret.inplace_addition(shortcut.forward(x));
    }
    else
    {
        ret.inplace_addition(x);
    }
    PRINT_TIMING("shortcut", my_timer.elapsed_time(true));

    // Remark: apply ReLU here! (found in debug)
    relu2.forward_inplace(ret);
    PRINT_TIMING("relu2", my_timer.elapsed_time(true));

    return ret;
}


void BasicBlock::load_from(ifstream& in_file)
{
    if (use_shortcut)
        shortcut.load_from(in_file);
    conv1.load_from(in_file);
    bn1.load_from(in_file);
    conv2.load_from(in_file);
    bn2.load_from(in_file);
}


DuplicateBasicBlocks::DuplicateBasicBlocks(int n_repeat, int in_channels, int out_channels)
{
    PRINT_CONSTRUCTOR("DuplicateBasicBlocks constructor\n");
    basic_blocks.reserve(n_repeat); // memory throughput optimization: avoid re-allocation

    if (in_channels != out_channels)
    {
        basic_blocks.push_back(BasicBlock(in_channels, out_channels));
    }
    
    while (basic_blocks.size() != (size_t)n_repeat)
    {
        basic_blocks.push_back(BasicBlock(out_channels, out_channels));
    }
}


DuplicateBasicBlocks::~DuplicateBasicBlocks()
{
    // PRINT_DEBUG("DuplicateBasicBlocks destructor\n");
}


TypedTensor DuplicateBasicBlocks::forward(const TypedTensor& x) const
{
    if (basic_blocks.size() < 1)
    {
        return x;
    }

    const BasicBlock& first_block = basic_blocks[0];

    TypedTensor ret = first_block.forward(x);

    for (auto it = basic_blocks.begin() + 1; it != basic_blocks.end(); ++ it)
    {
        ret = it->forward(ret);
    }
    return ret;
}


void DuplicateBasicBlocks::load_from(ifstream& in_file)
{
    for (BasicBlock& block: basic_blocks)
    {
        block.load_from(in_file);
    }
}


GlobalAveragePool2d::GlobalAveragePool2d()
{
    PRINT_CONSTRUCTOR("GlobalAveragePool2d constructor\n");
}


GlobalAveragePool2d::~GlobalAveragePool2d()
{
    // PRINT_DEBUG("GlobalAveragePool2d destructor\n");
}


TypedTensor GlobalAveragePool2d::forward(const TypedTensor& x) const
{   // x: (N, C, H, W)
    // output: (N, C)
    const vector<int>& x_shape = x.get_shape();
    int batch_size = x_shape[0];
    int n_channels = x_shape[1];
    int h_in = x_shape[2];
    int w_in = x_shape[3];

    TypedTensor ret(2, batch_size, n_channels);
    DTYPE *ret_ptr = ret.get_pointer();
    DTYPE *x_ptr = x.get_pointer();

    for (int n_sample = 0; n_sample < batch_size; ++ n_sample)
    {
        for (int c = 0; c < n_channels; ++ c)
        {
            ret_ptr[(n_sample * n_channels) + c] = 0.0;
            for (int i = 0; i < h_in; ++ i)
            {
                for (int j = 0; j < w_in; ++ j)
                {
                    ret_ptr[(n_sample * n_channels) + c] += //x.at(n_sample, c, i, j);
                        x_ptr[(((n_sample * n_channels) + c) * h_in + i) * w_in + j];
                }
            }
            ret.at(n_sample, c) /= (h_in * w_in);
        }
    }

    return ret;
}


void GlobalAveragePool2d::load_from(ifstream& in_file)
{
    // nothing to load
}


Softmax::Softmax()
{
    PRINT_CONSTRUCTOR("Softmax constructor\n");
}


Softmax::~Softmax()
{
    // PRINT_DEBUG("Softmax destructor\n");
}


TypedTensor Softmax::forward(const TypedTensor& x) const
{   // x: (N, F)
    // output: (N, F)
    
    // first do an element-wise exponential
    TypedTensor exp_x(x); 
    DTYPE *exp_x_ptr = exp_x.get_pointer();
    for (int i = 0; i < exp_x.get_total_elems(); ++ i)
    {
        exp_x_ptr[i] = exp(exp_x_ptr[i]);
    }

    // then divide each element by row-wise sum
    const vector<int>& x_shape = x.get_shape();
    int batch_size = x_shape[0];
    int num_feats = x_shape[1];
    for (int i = 0; i < batch_size; ++ i)
    {  
        // evaluate row-wise sum
        DTYPE rowwise_sum = exp_x.at(i, 0);
        for (int j = 1; j < num_feats; ++ j)
        {
            rowwise_sum += exp_x.at(i, j);
        }

        // division
        for (int j = 0; j < num_feats; ++ j)
        {
            exp_x.at(i, j) /= rowwise_sum;
        }
    }

    return exp_x;
}


void Softmax::load_from(ifstream& in_file)
{
    // nothing to load
}


ResNet34::ResNet34(int init_channel):
    network({
        new Conv2d(3, init_channel, {7, 7}, 3, 2),
        new BatchNorm2d(init_channel),
        new ReLU(),
        new MaxPool2d(3, 2, 1),
        new DuplicateBasicBlocks(3, init_channel, init_channel),
        new DuplicateBasicBlocks(4, init_channel, init_channel * 2),
        new DuplicateBasicBlocks(6, init_channel * 2, init_channel * 4),
        new DuplicateBasicBlocks(3, init_channel * 4, init_channel * 8),
        new GlobalAveragePool2d(),
        new Linear(init_channel * 8, 43),
        new Softmax()
    })
{
    PRINT_CONSTRUCTOR("ResNet34 constructor\n");
}


ResNet34::~ResNet34()
{
    PRINT_CONSTRUCTOR("ResNet34 destructor\n");
    for (InferenceBase* infer_module: network)
    {
        delete infer_module;
    }
}


TypedTensor ResNet34::forward(const TypedTensor& x) const
{
    MyTimer global_timer;
    int i = 1;  // module counter
    MyTimer my_timer;

    if (DEBUG_FLAG) printf("[resnet] forwarding in module 0 ...\n");
    TypedTensor ret = network[0]->forward(x);
    if (DEBUG_FLAG && TIMING_FLAG) 
            printf("[resnet] module 0 takes %.4f secs\n", my_timer.elapsed_time(true));
    if (ret.get_shape()[1] == 10 && DEBUG_NO1_FLAG)
    {
        ofstream debug_out("debug_out.txt", ios::out);
        debug_out << ret << endl;
        debug_out.close();

        exit(1); 
    }

    for (auto it = network.begin() + 1; it != network.end(); ++ it)
    {
        if (DEBUG_FLAG) printf("[resnet] forwarding in module %d ...\n", i);
        if ((* it)->get_inplace())
        {
            (* it)->forward_inplace(ret);
        }
        else
        {
            ret = (* it)->forward(ret);
        }
        // cout << ret << endl;
        if (DEBUG_FLAG && TIMING_FLAG) 
            printf("[resnet] module %d takes %.4f secs\n", i, my_timer.elapsed_time(true));
        ++ i;
    }
    PRINT_TIMING("ResNet34", global_timer.elapsed_time());

    return ret;
}


void ResNet34::load_from(ifstream& in_file)
{
    // network[0]->load_from(in_file);
    // network[1]->load_from(in_file);
    // network[2]->load_from(in_file);
    // network[3]->load_from(in_file);
    // network[4]->load_from(in_file);

    // cout << ((DuplicateBasicBlocks *)network[4])->basic_blocks[2].conv2.weight << endl;

    int i = 0; // module counter

    for (InferenceBase* infer_module: network)
    {
        infer_module->load_from(in_file);
        if (DEBUG_FLAG) printf("module %d loaded\n", i ++);
    }

    PRINT_DEBUG("ResNet34 successfully loaded\n");

    // cout << ((Linear *)network[9])->mbias << endl;
}



// int main(int argc, char *argv[])
// {
//     // Test ResNet34 checker model
//     double alpha = 0.15;

//     cout << "initiating model ..." << endl;

//     ResNet34 checker_model((int)round(64 * alpha));

//     cout << "model initiated" << endl;

//     ifstream in_file;
//     in_file.open("../model/traffic/best.checker.txt", ios::in);
//     checker_model.load_from(in_file);
//     in_file.close();

//     // TypedTensor x(4, 2, 3, 10, 12);
//     // x.load_from("./tensors/x.txt");
//     // // cout << "X:\n" << x << endl;

//     // Conv2d conv_layer(3, 4, {4, 5});
//     // conv_layer.load_weight_from("./tensors/kernel.txt");
//     // // cout << "\n\nconv.weight:\n" << conv_layer.weight << endl;

//     // x = conv_layer.forward(x);
//     // // cout << "\n\nconv(X):\n" << x << endl;

//     // // compare results with pytorch
//     // TypedTensor x_true(4, 2, 4, 7, 8);
//     // x_true.load_from("./tensors/conv_res.txt");
//     // cout << "\n\nConv diff:\n" << x - x_true << endl;

//     // MaxPool2d maxpool_layer(3, 2);
    
//     // x = maxpool_layer.forward(x);
//     // // cout << "\n\nmaxpool(X):\n" << x << endl;

//     // TypedTensor x1_true(4, 2, 4, 3, 3);
//     // x1_true.load_from("./tensors/maxpool_res.txt");
//     // cout << "\n\nMaxpool diff:\n" << x - x1_true << endl;

//     // TypedTensor x(4, 1, 2, 3, 3);
//     // double *arr = x.get_pointer();
//     // arr[0] = 1.;
//     // arr[4] = 2.;
//     // arr[8] = 3.;
//     // arr[11] = 4.;
//     // arr[13] = 5.;
//     // arr[15] = 6.;
//     // cout << "X:\n" << x << endl;
//     // vector<int> kernel_size({2, 2});
//     // //cout << unfoldTensor(x, kernel_size, 0, 1) << endl;

//     // // printf("hello myResNet!\n");
    
//     // Conv2d conv(2, 3, kernel_size, 0, 1, false);
//     // double *kernel = conv.weight.get_pointer();
//     // kernel[0] = kernel[1] = kernel[2] = kernel[3] = 1.;
//     // kernel += 4;
//     // kernel[0] = kernel[1] = kernel[2] = kernel[3] = 0.5;
//     // kernel += 4;
//     // kernel[0] = kernel[1] = kernel[2] = kernel[3] = 2.;

//     // cout << "W:\n" << conv.weight << endl;

//     // x = conv.forward(x);
//     // cout << "CONV:\n" << x << endl;

//     // MaxPool2d maxpool2d(2);
//     // x = maxpool2d.forward(x);
//     // cout << "MAXPOOL:\n" << x << endl;
    
//     return 0;
// }