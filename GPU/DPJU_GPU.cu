
#include <cuda.h>
#include <iostream>
#include <vector>

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdio.h>
#include <assert.h>
#include <windows.h>

#define TILE_W      16
#define TILE_H      16
#define RADIUS      0
#define EPS         0.1
#define DIAM        (RADIUS*2+1)
#define SIZE        (DIAM*DIAM)
#define BLOCK_W     (TILE_W+(2*RADIUS))
#define BLOCK_H     (TILE_H+(2*RADIUS))
#define PAR_NUM     8

using namespace cv;
using namespace std;

std::mutex mtx;
std::condition_variable produce, consume;
std::queue<cv::Mat> q;
int max_size = 20;
bool finished = false, inited = false;
int width, height, fps, frame_Number;

Mat m_src;
__device__ __managed__ int m_realk{ -1 };  //  The number of seed points obtained by calculation
__device__ __managed__ int m_step{ -1 };   //  Distance between seed points. By default,is equal to SigmaS(spacial kernal)
__device__ __managed__ int m_cols{ -1 };   //  The number of seed point cols by calculation
__device__ __managed__ int m_rows{ -1 };   //  The number of seed point rows by calculation

__device__ __managed__ uint m_count{ 0 };  //  The actual number of seed points was obtained through statistics
__device__ __managed__ uint m_r_cols{ 0 };  //  Actual number of seed point cols 
__device__ __managed__ uint m_r_rows{ 0 };  //  Actual number of seed point rows

__device__ __managed__ double m_sigmaR{ -1 };  // sigma of range kernal

__device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 operator+(float3 a, float b){
    return make_float3(a.x+b, a.y+b, a.z+b);
}
__device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 operator*(float3 a, float3 b){
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float3 operator/(float3 a, float3 b){
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
__device__ float3 operator/(float3 a, int n){
    return make_float3(a.x/(float)n, a.y/(float)n, a.z/(float)n);
}
__device__ float3 fmaxf3(float3 a, float b){
    return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

__device__ void box_filter(float3 *in, float3 *out, int width, int height)
{
    __shared__ float3 smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);
    const int idx = y * width + x;


    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = in[idx];
    __syncthreads();

    // float3 sum = make_float3(0, 0, 0);
    float3 sum = make_float3(0,0,0);
    int count = 0;
 
    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
            (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
       for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float3 i = smem[bindex + (dy * blockDim.x) + dx];
                sum = sum + i;
                ++count;
            }
        }
        out[idx] = sum / count;
    }
}

__device__ void compute_cov_var(float3 *mean_Ip, float3 *mean_II, float3 *mean_I,
        float3 *mean_p, float3 *var_I, float3 *cov_Ip, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 m_I = mean_I[idx];
    var_I[idx] = fmaxf3(mean_II[idx] - m_I * m_I, 0.);
    cov_Ip[idx] = fmaxf3(mean_Ip[idx] - m_I * mean_p[idx], 0.);
}

__device__ void compute_ab(float3 *var_I, float3 *cov_Ip, float3 *mean_I,
        float3 *mean_p, float3 *a, float3 *b, float eps, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float3 *in, float3 *mean_a, float3 *mean_b, float3 *q,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float3* d_input,
        float3 *d_p,
        float3 *d_q,
        float3 *mean_I,
        float3 *mean_p,
        float3 *mean_Ip,
        float3 *mean_II,
        float3 *var_I,
        float3 *cov_Ip,
        float3 *a,
        float3 *b,
        float3 *d_tmp,
        float3 *d_tmp2,
        float3 *mean_a,
        float3 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    box_filter(d_input, mean_I, width, height);
    box_filter(d_p, mean_p, width, height);
    __syncthreads();
    box_filter(d_tmp, mean_Ip, width, height);
    box_filter(d_tmp2, mean_II, width, height);
    if (x >= 0 && y >= 0 && x < width && y < height) {
    compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);
    compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    }
}

__global__ void output_kernel(float3* d_input,
        float3 *d_p,
        float3 *d_q,
        float3 *a,
        float3 *b,
        float3 *mean_a,
        float3 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    box_filter(a, mean_a, width, height);
    __syncthreads();
    box_filter(b, mean_b, width, height);
    __syncthreads();

    if (x >= 0 && y >= 0 && x < width && y < height) {
    compute_q(d_p, mean_a, mean_b, d_q, width, height);
    }
}

__global__ void mul_kernel(float3 *d_input, 
                    float3 *d_p,
                    float3 *d_tmp, 
                    float3 *d_tmp2,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_tmp[idx] = d_input[idx] * d_p[idx];
        d_tmp2[idx] = d_input[idx] * d_input[idx];
    }
}

__global__ void convert2float_kernel(uchar3 *d_input,
                    float3 *d_output,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_output[idx].x = d_input[idx].x/255.0;
        d_output[idx].y = d_input[idx].y/255.0;
        d_output[idx].z = d_input[idx].z/255.0;
    }
}

__global__ void convert2uchar_kernel(float3 *d_input,
                    uchar3 *d_output,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_output[idx].x = (uchar)(d_input[idx].x*255.0);
        d_output[idx].y = (uchar)(d_input[idx].y*255.0);
        d_output[idx].z = (uchar)(d_input[idx].z*255.0);
    }
}

class GuidedFilter{
    float3 *d_input, *d_p, *d_output, *d_mean_I, *d_mean_p, *d_mean_Ip,
           *d_mean_II, *d_var_I, *d_cov_Ip, *d_a, *d_b, *d_mean_a,
           *d_mean_b, *d_tmp, *d_tmp2;
    uchar3 *d_image_input, *d_image_p, *d_image_output;
    int height, width, m, n;
    public:
    GuidedFilter(int w, int h){
        width = w;
        height = h;
        m = width * height * sizeof(uchar3);
        n = width * height * sizeof(float3);

        cudaMalloc<uchar3>(&d_image_input, m);
        cudaMalloc<uchar3>(&d_image_p, m);
        cudaMalloc<uchar3>(&d_image_output, m);

        cudaMalloc<float3>(&d_input, n);
        cudaMalloc<float3>(&d_p, n);
        cudaMalloc<float3>(&d_output, n);
        cudaMalloc<float3>(&d_mean_I, n);
        cudaMalloc<float3>(&d_mean_p, n);
        cudaMalloc<float3>(&d_mean_Ip, n);
        cudaMalloc<float3>(&d_mean_II, n);
        cudaMalloc<float3>(&d_var_I, n);
        cudaMalloc<float3>(&d_cov_Ip, n);
        cudaMalloc<float3>(&d_a, n);
        cudaMalloc<float3>(&d_b, n);
        cudaMalloc<float3>(&d_mean_a, n);
        cudaMalloc<float3>(&d_mean_b, n);
        cudaMalloc<float3>(&d_tmp, n);
        cudaMalloc<float3>(&d_tmp2, n);
    }
    ~GuidedFilter(){
        cudaFree(d_image_input);
        cudaFree(d_image_output);
        cudaFree(d_image_p);
        cudaFree(d_input);
        cudaFree(d_p);
        cudaFree(d_output);
        cudaFree(d_mean_I);
        cudaFree(d_mean_p);
        cudaFree(d_mean_Ip);
        cudaFree(d_mean_II);
        cudaFree(d_var_I);
        cudaFree(d_cov_Ip);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_mean_a);
        cudaFree(d_mean_b);
        cudaFree(d_tmp);
        cudaFree(d_tmp2);
    }
    void filter(uchar3* image_input, uchar3* image_output, uchar3* image_p, cudaStream_t stream)
    {
        int GRID_W = ceil(width /(float)TILE_W);
        int GRID_H = ceil(height / (float)TILE_H);

        const dim3 block(BLOCK_W, BLOCK_H);
        const dim3 grid(GRID_W, GRID_H);

        cudaMemcpyAsync(d_image_input, image_input, width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_image_p, image_p, width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
        convert2float_kernel<<<grid, block, 0, stream>>>(d_image_input, d_input, width, height);
        convert2float_kernel<<<grid, block, 0, stream>>>(d_image_p, d_p, width, height);

        mul_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_tmp, d_tmp2, width, height);

        mean_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_output, d_mean_I, d_mean_p, d_mean_Ip,d_mean_II, d_var_I, d_cov_Ip, d_a, d_b, d_tmp, d_tmp2, d_mean_a,
        d_mean_b, width, height, EPS);

        output_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_output, d_a, d_b,d_mean_a, d_mean_b, width, height, EPS);
        convert2uchar_kernel<<<grid, block, 0, stream>>>(d_output, d_image_output, width, height);
        cudaMemcpyAsync(image_output, d_image_output, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);
    }
};

void consumer(std::string output_file){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter writer;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    while(!inited);
    writer.open(output_file.c_str(), codec, fps, cv::Size(width, height), true);
    auto guidedfilter = GuidedFilter(width, height);
    cv::Mat input(height, width, CV_8UC3), output(height, width, CV_8UC3);

    while(!finished){
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size()==0)  consume.wait(lck);
        input = q.front();
        q.pop();
        guidedfilter.filter(input.ptr<uchar3>(), output.ptr<uchar3>(), input.ptr<uchar3>(), stream);
        writer.write(output);
        
        produce.notify_all();
        lck.unlock();
    }
    writer.release();
}

void producer(std::string input_file){
    cv::VideoCapture capture(input_file.c_str());
    if(!capture.isOpened()){
        std::cout<<"could not open the video!\n";
    }
    fps = capture.get(cv::CAP_PROP_FPS);
    height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_Number = capture.get(cv::CAP_PROP_FRAME_COUNT);
    inited = true;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));
    cv::Mat frame(height, width, CV_8UC3);
    while(capture.read(frame)){
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size()==max_size)   produce.wait(lck);
        q.emplace(frame);
        consume.notify_all();
        lck.unlock();
    }
    finished = true;

    capture.release();
}



// CUDA 错误处理宏
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void getColorWeightBuffer(double* CWBuffer, double sigma)
{
    double temp_ = -0.5 / (sigma * sigma);
    double temp_2 = 0;
    int i = threadIdx.x;
    CWBuffer[i] = exp(i * i * temp_);
    temp_2 = 1.0 / powf(2, i);
    if (temp_2 > CWBuffer[i])
        CWBuffer[i] = temp_2;
}

__global__ void upForchannels3(int rows, int cols, int src_size, double* weight_count, double* gray_weight, cuda::PtrStepSz<uchar3> udst)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < src_size)
    {
        int col = i % cols;
        int row = i / cols;
        udst(row, col) = make_uchar3(gray_weight[i * 3] / weight_count[i], gray_weight[i * 3 + 1] / weight_count[i], gray_weight[i * 3 + 2] / weight_count[i]);
    }
    else return;
}

__global__ void upForUp3(uint m_r_cols, int m_step, cuda::PtrStepSz<uchar3> m_src, cuda::PtrStepSz<uchar3> vect, int r, int offset, int rows, int cols, int ksize, int channels, double* CWBuffer, double* weight_count, double* gray_weight, cuda::PtrStepSz<uchar3> src_data)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int l = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < m_count)
    {
        int row = (x / m_r_cols) * m_step + offset;
        int col = (x % m_r_cols) * m_step + offset;
        if (row <= rows && col <= cols)
        {
            int LT = row - r;
            int RT = col - r;
            int LB = row - r - m_src.rows + 1;
            int RB = col - r - m_src.cols + 1;

            int filterImg_row = x / m_r_cols;
            int filterImg_col = x % m_r_cols;
            auto filterImg_data = vect;

            if (k < ksize && l < ksize && LT + k >= 0 && RT + l >= 0 && LB <= -k && RB <= -l)
            {
                int gray_index = ((LT + k) * cols + RT + l);
                int gray_row = gray_index / cols;
                int gray_col = gray_index % cols;
                double w = CWBuffer[abs(src_data(row, col).x - src_data(gray_row, gray_col).x)] *
                    CWBuffer[abs(src_data(row, col).y - src_data(gray_row, gray_col).y)] *
                    CWBuffer[abs(src_data(row, col).z - src_data(gray_row, gray_col).z)];
                gray_weight[((LT + k) * cols + RT + l) * 3] += w * filterImg_data(filterImg_row, filterImg_col).x;
                gray_weight[((LT + k) * cols + RT + l) * 3 + 1] += w * filterImg_data(filterImg_row, filterImg_col).y;
                gray_weight[((LT + k) * cols + RT + l) * 3 + 2] += w * filterImg_data(filterImg_row, filterImg_col).z;

                weight_count[(LT + k) * cols + RT + l] += w;
            }
            else
            {
                return;
            }
        }
        else
        {
            return;
        }
    }
    else
    {
        return;
    }
}

void getBaseData(const Mat& src, uint ksize)
{
    src.copyTo(m_src);
    m_step = static_cast<int>(sqrt(m_src.cols * m_src.rows / ksize));

    if (fmod(m_src.cols, m_step) > m_step / 2) {
        m_r_cols = m_cols = m_src.cols / m_step + 1;
        m_r_rows = m_rows = m_src.rows / m_step + 1;
    }
    else {
        m_r_cols = m_cols = m_src.cols / m_step;
        m_r_rows = m_rows = m_src.rows / m_step;
    }
    m_realk = m_cols * m_rows;

    int offset = m_step / 2;
    int rows = m_src.rows;
    int cols = m_src.cols;
    m_count = 0;

    for (int i = 0; i < m_realk; i++) {
        int row = (i / m_cols) * m_step + offset;
        if (row > rows) break;
        int col = (i % m_cols) * m_step + offset;
        if (col > cols) continue;
        int index = (row * cols + col);
        m_count++;
    }

    if (m_count == m_realk)
        return;
    if (m_count != m_realk && m_realk - m_count == m_rows) {
        m_r_cols -= 1;
    }
    else if (m_realk - m_count == m_cols) {
        m_r_rows -= 1;
    }
    else {
        std::cerr << "种子点选取错误" << std::endl;
        exit(-1);
    }
}

void upSampling(Mat& dst, const Mat* vect)
{
    int r = 2 * m_step;
    int cols = m_src.cols;
    int rows = m_src.rows;
    int ksize = 2 * r + 1;
    int channels = m_src.channels();
    int src_size = cols * rows;
    double* CWBuffer;
    cudaMallocManaged((void**)&CWBuffer, 256 * sizeof(double));

    getColorWeightBuffer << <1, 256>> > (CWBuffer, m_sigmaR);
    checkCuda(cudaDeviceSynchronize());

    double* gray_weight;
    double* weight_count;
    cudaMallocManaged((void**)&gray_weight, src_size * channels * sizeof(double));
    cudaMallocManaged((void**)&weight_count, src_size * sizeof(double));

    int offset = m_step / 2;

    //up speed with gpu
    int nthreads = 16;
    dim3 block(nthreads, 1, 1);
    int nblocks1 = (m_count + nthreads + 1) / nthreads;
    int nblocks2 = (src_size + nthreads + 1) / nthreads;
    dim3 grid(nblocks1, ksize, ksize);
    cuda::GpuMat vectGpu;
    vectGpu.upload(*vect);
    cuda::GpuMat m_src_gpu;
    m_src_gpu.upload(m_src);
    int channels_gpu;
    cudaMallocManaged((void**)&channels_gpu, sizeof(int));
    channels_gpu = channels;
    int ksize_gpu;
    cudaMallocManaged((void**)&ksize_gpu, sizeof(int));
    ksize_gpu = ksize;

    upForUp3 << <grid, block>> > (m_r_cols, m_step, m_src_gpu, vectGpu, r, offset, rows, cols, ksize_gpu, channels_gpu, CWBuffer, weight_count, gray_weight, m_src_gpu);
    checkCuda(cudaDeviceSynchronize());

    cuda::GpuMat tdst(rows, cols, CV_8UC3);
    upForchannels3 << <nblocks2, nthreads>> > (rows, cols, src_size, weight_count, gray_weight, tdst);
    checkCuda(cudaDeviceSynchronize());
    tdst.download(dst);

    cudaFree(CWBuffer);
    cudaFree(gray_weight);
    cudaFree(weight_count);
    cudaFree(&m_src_gpu);
    cudaFree(&vectGpu);
}

int main() {
    string input_path = "LR.tif";
    string guide_path = "Guide.tif";
    Mat src1 = imread(input_path, IMREAD_ANYCOLOR);
    Mat src2 = imread(guide_path, IMREAD_ANYCOLOR);
    Mat matrix_trans_a;
    src1.convertTo(matrix_trans_a, CV_64FC3);
    width = src2.cols;
    height = src2.rows;
    Mat out1;
    Mat out2(height, width, CV_8UC3);
    int ksize = src1.cols * src1.rows;
    CV_DbgAssert(src1.channels() == 1 || src1.channels() == 3);
    CV_DbgAssert(src2.channels() == 1 || src2.channels() == 3);
    auto guidedfilter = GuidedFilter(width, height);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    getBaseData(src2, ksize);
    upSampling(out1, &src1);
    guidedfilter.filter(src2.ptr<uchar3>(), out2.ptr<uchar3>(), out1.ptr<uchar3>(), stream);

    imshow("Output", out2);
    imwrite("Output.png", out2);
    waitKey(0);
    return 0;
}
