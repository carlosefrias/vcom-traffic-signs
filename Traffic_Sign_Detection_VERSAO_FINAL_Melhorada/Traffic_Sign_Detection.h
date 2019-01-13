#include <atlstr.h>
#include <math.h>
#include <sstream>
#include <iostream> // std::cout
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // cv::imread()
#include <opencv2/imgproc/imgproc.hpp> // cv::Canny

#define IMG 0
#define VDO 1
#define RED 2 //#FF0000
#define BLACK 3 //#000000
#define TEMPLATE_HEIGHT 165
#define MIN_SIGN_HEIGHT 10
#define MIN_CORRELATION_ACCEPTED 0.80
#define MAX_NOISE_ACCEPTED 0.05
#define BW_THRESHOLD 100 // range from 0 to 255
#define DEBUG_IMG false
#define DEBUG_IMG_CROP true
#define DEBUG_TEX false
#define CARLOS_COMPUTER true

struct Size_number{
	int pointX;
	int pointY;
	int width;
	int height;
} size_number;

using namespace cv;
using namespace std;

Size_number get_size_numbers(Mat &img, Mat &img_mask_thres);
double matchingMethod(Mat &img, Mat &templ, int signalToDetect);
vector<Mat> crop_sign(Mat &src, Mat &img_bw, vector<Vec3f> &circles);
bool hough_detection(Mat &image, vector<Vec3f> &circles, Mat &img_hough);
bool detect_color(Mat &srcImageNameNotEqualized, Mat &srcImageNameEqualized, Mat &output_img_bw, Mat &output_img_eq_bw);
bool detect_mask(Mat &imageinput, Mat &imagemask, int color);
Mat image_hist_equalizer(Mat inputimage);
bool menu(string &image);
bool detect_files(int type_file, string &filename);
bool crop_video(string video, string image);
bool crop_stream(string image);
void rotateImage(Mat &image);
Mat histogramImage(Mat &image);
