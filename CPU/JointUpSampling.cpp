#pragma once
#include <opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <mex.h>
using namespace cv;
using namespace Eigen;
using namespace std;

class BaseBilateralFilter
{

protected:
	BaseBilateralFilter();

	virtual void init(const Mat& src1, const Mat& src2, int step);
	virtual void upSampling(int r_num, Mat& dst, const Mat* vect, const Mat* guid = nullptr);
	inline void getColorWeightBuffer(double* CWBuffer);

protected:

	Mat m_src;

	int m_step{ -1 };

	int channels_1 = { -1 };
	int channels_2 = { -1 };

	uint m_count{ 0 };  
	uint m_r_cols{ 0 };  
	uint m_r_rows{ 0 };  
};

class Filter3 :
	public BaseBilateralFilter
{
public:
	Filter3();

	void filter(Mat& src1, Mat& src2, int r_num, int step);
	Mat dst;
};

Filter3::Filter3()
{
}

void Filter3::filter(Mat& src1, Mat& src2, int r_num, int step)
{
	CV_DbgAssert(src1.channels() == 1 || src1.channels() == 3);
	CV_DbgAssert(src2.channels() == 1 || src2.channels() == 3);
	channels_1 = src1.channels();
	channels_2 = src2.channels();
	init(src1, src2, step);
	Mat matrix_trans_a;
	src1.convertTo(matrix_trans_a, CV_64FC3);
	upSampling(r_num, this->dst, &matrix_trans_a);
}

typedef Eigen::Triplet<double> TD;

BaseBilateralFilter::BaseBilateralFilter()
{
}

void BaseBilateralFilter::init(const Mat& src1, const Mat& src2, int step)
{
	src2.copyTo(m_src);
	m_r_cols = src1.cols;
	m_r_rows = src1.rows;
	m_step = step;
}

void BaseBilateralFilter::upSampling(int r_num, Mat& dst, const Mat* vect, const Mat* guid)
{
	if (guid == nullptr) {
		guid = &m_src;
	}

	int r = r_num * m_step;
	int cols = m_src.cols;
	int rows = m_src.rows;
	int ksize = 2 * r + 1;
	int src_size = cols * rows;
	auto src_data = m_src.data;
	auto guid_data = guid->data;
	double* CWBuffer = new double[256]{ 0 };
	getColorWeightBuffer(CWBuffer);
	double* gray_weight = new double[src_size * channels_1]{ 0 };

	double* weight_count = new double[src_size] {0};


	int offset = m_step / 2;
	m_count = m_r_cols * m_r_rows;
	
	int count = 0;
	int num = 8;
#pragma omp parallel for num_threads(num)
	for (int i = 0; i < m_count; i++) {
		int row = (i / m_r_cols) * m_step + offset;
		if (row > rows) break;
		int col = (i % m_r_cols) * m_step + offset;
		if (col > cols) continue;

		int index = (row * cols + col) * channels_2;

		int LT = row - r;
		int RT = col - r;
		int LB = row - r - m_src.rows + 1;
		int RB = col - r - m_src.cols + 1;

		int filterImg_row = i / m_r_cols;
		int filterImg_col = i % m_r_cols;

		auto filterImg_data = vect->ptr<double>(filterImg_row);
		for (int k = 0; k < ksize; k++) {
			for (int l = 0; l < ksize; l++) {
				if (LT + k < 0 || RT + l < 0 || LB > -k || RB > -l) continue;
				if (channels_1 == 1 && channels_2 == 1) {
					uchar gray = src_data[(LT + k) * cols + RT + l];
					double w = CWBuffer[abs(src_data[index] - gray)];
					gray_weight[(LT + k) * cols + RT + l] += w * filterImg_data[filterImg_col];
					weight_count[(LT + k) * cols + RT + l] += w;
				}
				else
				{
					if (channels_1 == 3 && channels_2 == 3) {
						int gray_index = ((LT + k) * cols + RT + l) * 3;
						double w = CWBuffer[abs(src_data[index] - src_data[gray_index])] *
							CWBuffer[abs(src_data[index + 1] - src_data[gray_index + 1])] *
							CWBuffer[abs(src_data[index + 2] - src_data[gray_index + 2])];

						gray_weight[((LT + k) * cols + RT + l) * 3] += w * filterImg_data[filterImg_col * 3];
						gray_weight[((LT + k) * cols + RT + l) * 3 + 1] += w * filterImg_data[filterImg_col * 3 + 1];
						gray_weight[((LT + k) * cols + RT + l) * 3 + 2] += w * filterImg_data[filterImg_col * 3 + 2];

						weight_count[(LT + k) * cols + RT + l] += w;
					}
					else
					{
						if (channels_1 == 1 && channels_2 == 3) {
							int gray_index = ((LT + k) * cols + RT + l) * 3;
							double w = CWBuffer[abs(src_data[index] - src_data[gray_index])] *
								CWBuffer[abs(src_data[index + 1] - src_data[gray_index + 1])] *
								CWBuffer[abs(src_data[index + 2] - src_data[gray_index + 2])];

							gray_weight[(LT + k) * cols + RT + l] += w * filterImg_data[filterImg_col];
							weight_count[(LT + k) * cols + RT + l] += w;
						}
						else
						{
							uchar gray = src_data[(LT + k) * cols + RT + l];
							double w = CWBuffer[abs(src_data[index] - gray)];

							gray_weight[((LT + k) * cols + RT + l) * 3] += w * filterImg_data[filterImg_col * 3];
							gray_weight[((LT + k) * cols + RT + l) * 3 + 1] += w * filterImg_data[filterImg_col * 3 + 1];
							gray_weight[((LT + k) * cols + RT + l) * 3 + 2] += w * filterImg_data[filterImg_col * 3 + 2];

							weight_count[(LT + k) * cols + RT + l] += w;
						}
					}
				}
			}
		}
		count++;
	}

	if (channels_1 == 3) {
		Mat tdst(rows, cols, CV_8UC3);
		uchar* udst = tdst.data;
#pragma omp parallel for num_threads(num)
		for (int i = 0; i < src_size; i++) {
			udst[i * 3] = gray_weight[i * 3] / weight_count[i];
			udst[i * 3 + 1] = gray_weight[i * 3 + 1] / weight_count[i];
			udst[i * 3 + 2] = gray_weight[i * 3 + 2] / weight_count[i];
		}
		tdst.copyTo(dst);
	}
	else if (channels_1 == 1) {
		Mat tdst(rows, cols, CV_8UC1);
		uchar* udst = tdst.data;
#pragma omp parallel for num_threads(num)
		for (int i = 0; i < src_size; i++) {
			udst[i] = gray_weight[i] / weight_count[i];
		}
		tdst.copyTo(dst);
	}
	delete[] CWBuffer;
	delete[] gray_weight;
	delete[] weight_count;
}

inline void BaseBilateralFilter::getColorWeightBuffer(double* CWBuffer)
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < 256; i++) {
	      CWBuffer[i] = 1.0 / pow((i + 1), 3);
	}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs != 5) {
		mexErrMsgTxt("Five arguments are required.");
	}
    
    String src1_path = mxArrayToString(prhs[0]);
    String src2_path = mxArrayToString(prhs[1]);
    String out_path = mxArrayToString(prhs[2]);
    int r_num = mxGetScalar(prhs[3]);
	int step = mxGetScalar(prhs[4]);
    Mat src1 = imread(src1_path, IMREAD_ANYCOLOR);
    Mat src2 = imread(src2_path, IMREAD_ANYCOLOR);
	Filter3 filter; 
	double time = static_cast<double>(getTickCount());
	filter.filter(src1, src2, r_num, step);
	time = ((double)getTickCount() - time) / getTickFrequency();
	cout << "all times : " << time << endl;
	imwrite(out_path, filter.dst);
    double *out_pr ;
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    out_pr = mxGetPr(plhs[0]);
    *out_pr = time;
}