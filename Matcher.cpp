#include "Matcher.h"


Matcher::Matcher(const cv::Mat _matched_img, cv::Mat _mask)
{
	cv::Mat mask;
	if (!(_mask.empty()))
	{
		cv::cvtColor(_mask, mask, cv::COLOR_RGB2GRAY);
		cv::threshold(mask, mask, 175, 255, cv::THRESH_BINARY_INV);
		cv::resize(mask, mask, cv::Size(), resize_koef, resize_koef);
	}
	cv::cvtColor(_matched_img, matched_img, cv::COLOR_RGB2GRAY);
	cv::resize(matched_img, matched_img, cv::Size(), resize_koef, resize_koef);
	f2d = cv::ORB::create(number_of_points);
	f2d->detect(matched_img, matched_keypoints, mask);
	f2d->compute(matched_img, matched_keypoints, matched_descriptors);
}


Matcher::~Matcher()
{
	f2d->clear();
	matched_img.release();
	matched_keypoints.clear();
	matched_descriptors.release();
}


bool Matcher::match(const cv::Mat _input_img)
{
	//-- Working on input image
	cv::Mat gray;
	cv::cvtColor(_input_img, gray, cv::COLOR_RGB2GRAY);
	cv::resize(gray, gray, cv::Size(), resize_koef, resize_koef);

	//-- Compare hisograms
	int histSize = 8;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	cv::Mat matched_hist, hist;
	cv::calcHist(&matched_img, 1, 0, cv::Mat(), matched_hist, 1, &histSize, &histRange);
	cv::normalize(matched_hist, matched_hist, 0, 1, cv::NORM_MINMAX);
	cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
	double hist_compare = cv::compareHist(matched_hist, hist, cv::HISTCMP_CORREL);
	//-- If hist doesn't correlate
	if (hist_compare < hist_accuracy)
		return false;

	//-- Feature matching
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	f2d->detect(gray, keypoints);
	f2d->compute(gray, keypoints, descriptors);
	//-- If incorrect descriptors
	if (descriptors.type() != matched_descriptors.type() || descriptors.cols != matched_descriptors.cols)
		return false;
	cv::BFMatcher matcher;
	std::vector<cv::DMatch> matches, matches12, matches21;
	matcher.match(matched_descriptors, descriptors, matches12);
	matcher.match(descriptors, matched_descriptors, matches21);

	//-- Cross checking matches
	for (size_t i = 0; i < matches12.size(); i++)
	{
		cv::DMatch forward = matches12[i];
		cv::DMatch backward = matches21[forward.trainIdx];
		if (backward.trainIdx == forward.queryIdx)
			matches.push_back(forward);
	}
	if (matches.size() > 3) {
		//-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;
		for (int i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(matched_keypoints[matches[i].queryIdx].pt);
			scene.push_back(keypoints[matches[i].trainIdx].pt);
		}
		cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);

		if (H.empty())
		{
			return false;
		}

		double h11 = H.at<double>(0, 0);			
		double h12 = H.at<double>(0, 1);
		double h21 = H.at<double>(1, 0);
		double h22 = H.at<double>(1, 1);

		if (abs(h11 - 1) < 1.0 - homography_accuracy &&
			abs(h22 - 1) < 1.0 - homography_accuracy &&
			abs(h12) < 1.0 - homography_accuracy &&
			abs(h21) < 1.0 - homography_accuracy)		
		{
			return true;
		}
		return false;
	}
	return false;
}

