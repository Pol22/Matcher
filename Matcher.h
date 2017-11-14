#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


class Matcher
{
public:
	Matcher(cv::Mat _matched_img, cv::Mat _mask = cv::Mat());
	~Matcher();
	bool match(cv::Mat input_img);
private:
	// Model params
	double hist_accuracy = 0.8;
	double homography_accuracy = 0.8;
	double resize_koef = 1.0 / 8; // resize input images
	int number_of_points = 100; // number of features

	cv::Mat matched_img;
	cv::Ptr<cv::FeatureDetector> f2d;
	std::vector<cv::KeyPoint> matched_keypoints;
	cv::Mat matched_descriptors;
};