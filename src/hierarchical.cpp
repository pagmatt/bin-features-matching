#include "hierarchical.h"
// Salient points and features computation
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
// OpenCV
using namespace cv;

int const feat_to_compute = 100;
int const max_features_to_search = 25;
int const branching_factor = 5;
int const max_leaves_amount = 10;
int const trees_amount = 1;
int const top_k_feat = 2; // Extract just top 2, enough in order to use NNDR technique
int const px_to_draw = 100;
int const v_gap = 30;
int const h_gap = 50;
float const max_nndr_ratio = 0.9;
std::string ref_path = "../testing_dataset/img_ref.png";
std::string target_path = "../testing_dataset/img2.png"; // number can be set in [1,5]

cv::Mat
find_ORB_matches(cv::Mat &src, cv::Mat &dest);

cv::Mat
find_SIFT_matches(cv::Mat &src, cv::Mat &dest);

void profile_orb(cv::Mat &src, cv::Mat &dest, bool lin);

void profile_sift(cv::Mat &src, cv::Mat &dest);

int main(int, char **)
{

	// Loading of the image files

	cv::Mat src = cv::imread(samples::findFile(ref_path), IMREAD_GRAYSCALE);
	if (src.empty())
	{
		cout << "Could not open or find the reference image!\n"
			 << endl;
		return -1;
	}
	cv::Mat dest = cv::imread(samples::findFile(target_path), IMREAD_GRAYSCALE);
	if (dest.empty())
	{
		cout << "Could not open or find the target image!\n"
			 << endl;
		return -1;
	}

	// Compute matches
	cv::Mat orb_matches = find_ORB_matches(src, dest);
	cv::Mat sift_matches = find_SIFT_matches(src, dest);
	// Features computation : ORB comparison

	cv::imwrite("../orb_matches.jpeg", orb_matches);
	cv::imwrite("../sift_matches.jpeg", sift_matches);

	// Profile code
	//profile_orb(src, dest, true);
	//profile_sift(src, dest);
}

cv::Mat
find_ORB_matches(cv::Mat &src, cv::Mat &dest)
{
	Ptr<ORB> src_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint> src_orb_points;
	src_orb_obj->detect(src, src_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat src_out_orb_feat;
	src_orb_obj->compute(src, src_orb_points, src_out_orb_feat);
	//cout << "Computed " << out_orb_feat.size().height << " features!" << endl;

	Ptr<ORB> dest_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint> dest_orb_points;
	dest_orb_obj->detect(dest, dest_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat dest_out_orb_feat;
	dest_orb_obj->compute(dest, dest_orb_points, dest_out_orb_feat);

	// Output image
	cv::Mat stacked_orb = cv::Mat::zeros(px_to_draw * feat_to_compute + v_gap * (feat_to_compute - 1),
										 px_to_draw * 2 + h_gap, CV_8U); // Create stacked image canvas
	stacked_orb.setTo((cv::Scalar(255, 255, 255)));						 // Make it white

	unsigned valid_feat = 0; // Keeps track of how many valid features we have matched up to now
	unsigned skipped = 0;
	std::vector<unsigned> distances;

	// Search for similar features
	for (int j = 0; j < feat_to_compute; j++) // TODO: 5 -> feat-to-compute
	{

		cv::Mat out = MatchingLibs::parallel_search(dest_out_orb_feat, branching_factor, max_leaves_amount, trees_amount,
													max_features_to_search, top_k_feat, src_out_orb_feat.row(j));

		// Linear search, just for profiling purposes
		//cv::Mat out_lin = MatchingLibs::linear_search(dest_out_orb_feat, src_out_orb_feat.row(j));

		// Nearest Neighbour Distance Ratio (NNDR) to skim the matches and keep only the best ones
		unsigned dist_to_query = cv::norm(src_out_orb_feat.row(j), out.row(0), cv::NORM_HAMMING);
		if (dist_to_query / cv::norm(src_out_orb_feat.row(j), out.row(1), cv::NORM_HAMMING) < max_nndr_ratio)
		{
			continue;
		}

		// Visualize search results: target feature
		Point2f src_kp = src_orb_points[j].pt; // Get coord of src keypoint
		cv::Rect src_to_crop(src_kp.x - px_to_draw / 2, src_kp.y - px_to_draw / 2, px_to_draw, px_to_draw);
		// Bounds might be cross image, if so skip
		if (src_kp.x - px_to_draw / 2 < 0 || src_kp.y - px_to_draw / 2 < 0 ||
			src_kp.x + px_to_draw / 2 > src.rows || src_kp.y + px_to_draw / 2 > src.cols)
		{
			skipped++;
			continue;
		}
		cv::Mat src_crop = src(src_to_crop);

		// Visualize search results: feature match
		// Store the distance to the query
		distances.push_back(dist_to_query);
		// Must find the index in the original matrix, in order to associate feature to its descriptor
		Point2f dest_kp = dest_orb_points[MatchingLibs::search_feature(dest_out_orb_feat, out.row(0))].pt;	  // Get coord of matched, closest feature
		cv::Rect dst_to_crop(dest_kp.x - px_to_draw / 2, dest_kp.y - px_to_draw / 2, px_to_draw, px_to_draw); // x_start, y_start, width, height
		if (dest_kp.x - px_to_draw / 2 < 0 || dest_kp.y - px_to_draw / 2 < 0 ||
			dest_kp.x + px_to_draw / 2 > dest.rows || dest_kp.y + px_to_draw / 2 > dest.cols)
		{
			skipped++;
			continue;
		}
		cv::Mat dest_crop = dest(dst_to_crop);

		src_crop.copyTo(stacked_orb.colRange(0, px_to_draw).rowRange((v_gap + px_to_draw) * valid_feat, px_to_draw + (v_gap + px_to_draw) * valid_feat));
		dest_crop.copyTo(stacked_orb.colRange(px_to_draw + h_gap, 2 * px_to_draw + h_gap).rowRange((v_gap + px_to_draw) * valid_feat, px_to_draw + (v_gap + px_to_draw) * valid_feat));

		valid_feat++;
	}

	// Crop and keep only portion of the image that is actually used
	cv::Rect valid_crop(0, 0, stacked_orb.cols, valid_feat * (px_to_draw) + (valid_feat - 1) * v_gap); // x_start, y_start, width, height
	stacked_orb = stacked_orb(valid_crop);
	std::cout << valid_feat << " valid ORB matches found!" << std::endl;
	if (skipped > 0)
	{
		std::cout << "(skipped " << skipped << " features as their keypoins exceed image bounds)" << std::endl;
	}
	return stacked_orb;
}

cv::Mat
find_SIFT_matches(cv::Mat &src, cv::Mat &dest)
{
	Ptr<SIFT> src_sift_obj = SIFT::create(feat_to_compute);
	vector<KeyPoint> src_sift_points;
	src_sift_obj->detect(src, src_sift_points);
	//cout << "Detected " << src_sift_points.size() << " keypoints!" << endl;
	cv::Mat src_out_sift_feat;
	src_sift_obj->compute(src, src_sift_points, src_out_sift_feat);
	//cout << "Computed " << src_out_sift_feat.size().height << " features!" << endl;

	Ptr<SIFT> dest_sift_obj = SIFT::create(feat_to_compute);
	vector<KeyPoint> dest_sift_points;
	dest_sift_obj->detect(dest, dest_sift_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat dest_out_sift_feat;
	dest_sift_obj->compute(dest, dest_sift_points, dest_out_sift_feat);

	//Rows = #features, cols = # components, type is 32F
	//std::cout << CV_MAT_TYPE(dest_out_sift_feat.type()) << std::endl;
	//Quantize the features
	cv::Mat quantized_dest_sift = MatchingLibs::median_quantize(dest_out_sift_feat);
	cv::Mat quantized_src_sift = MatchingLibs::median_quantize(src_out_sift_feat);

	// Output image
	cv::Mat stacked_sift = cv::Mat::zeros(px_to_draw * feat_to_compute + v_gap * (feat_to_compute - 1),
										  px_to_draw * 2 + h_gap, CV_8U); // Create stacked image canvas
	stacked_sift.setTo((cv::Scalar(255, 255, 255)));					  // Make it white

	unsigned valid_feat = 0; // Keeps track of how many valid features we have matched up to now
	unsigned skipped = 0;
	std::vector<unsigned> distances;

	// Search for similar features
	for (int j = 0; j < feat_to_compute; j++) // TODO: 5 -> feat-to-compute
	{
		cv::Mat out = MatchingLibs::parallel_search(quantized_dest_sift, branching_factor, max_leaves_amount, trees_amount,
													max_features_to_search, top_k_feat, quantized_src_sift.row(j));

		// Nearest Neighbour Distance Ratio (NNDR) to skim the matches and keep only the best ones
		unsigned dist_to_query = cv::norm(quantized_dest_sift.row(j), out.row(0), cv::NORM_HAMMING);
		if (dist_to_query / cv::norm(quantized_dest_sift.row(j), out.row(1), cv::NORM_HAMMING) < max_nndr_ratio)
		{
			continue;
		}

		// Visualize search results: target feature
		Point2f src_kp = src_sift_points[j].pt; // Get coord of src keypoint
		cv::Rect src_to_crop(src_kp.x - px_to_draw / 2, src_kp.y - px_to_draw / 2, px_to_draw, px_to_draw);
		// Bounds might be cross image, if so skip
		if (src_kp.x - px_to_draw / 2 < 0 || src_kp.y - px_to_draw / 2 < 0 ||
			src_kp.x + px_to_draw / 2 > src.rows || src_kp.y + px_to_draw / 2 > src.cols)
		{
			skipped++;
			continue;
		}
		cv::Mat src_crop = src(src_to_crop);
		// Visualize search results: feature match
		// Store the distance to the query
		distances.push_back(dist_to_query);
		// Must find the index in the original matrix, in order to associate feature to its descriptor
		Point2f dest_kp = dest_sift_points[MatchingLibs::search_feature(quantized_dest_sift, out.row(0))].pt; // Get coord of matched, closest feature
		cv::Rect dst_to_crop(dest_kp.x - px_to_draw / 2, dest_kp.y - px_to_draw / 2, px_to_draw, px_to_draw); // x_start, y_start, width, height
		if (dest_kp.x - px_to_draw / 2 < 0 || dest_kp.y - px_to_draw / 2 < 0 ||
			dest_kp.x + px_to_draw / 2 > dest.rows || dest_kp.y + px_to_draw / 2 > dest.cols)
		{
			skipped++;
			continue;
		}
		cv::Mat dest_crop = dest(dst_to_crop);

		src_crop.copyTo(stacked_sift.colRange(0, px_to_draw).rowRange((v_gap + px_to_draw) * valid_feat, px_to_draw + (v_gap + px_to_draw) * valid_feat));
		dest_crop.copyTo(stacked_sift.colRange(px_to_draw + h_gap, 2 * px_to_draw + h_gap).rowRange((v_gap + px_to_draw) * valid_feat, px_to_draw + (v_gap + px_to_draw) * valid_feat));

		valid_feat++;
	}

	// Crop and keep only portion of the image that is actually used
	cv::Rect valid_crop(0, 0, stacked_sift.cols, valid_feat * (px_to_draw) + (valid_feat - 1) * v_gap); // x_start, y_start, width, height
	stacked_sift = stacked_sift(valid_crop);
	std::cout << valid_feat << " valid SIFT matches found!" << std::endl;
	if (skipped > 0)
	{
		std::cout << "(skipped " << skipped << " features as their keypoins exceed image bounds)" << std::endl;
	}

	return stacked_sift;
}

void profile_orb(cv::Mat &src, cv::Mat &dest, bool lin)
{
	Ptr<ORB> src_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint> src_orb_points;
	src_orb_obj->detect(src, src_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat src_out_orb_feat;
	src_orb_obj->compute(src, src_orb_points, src_out_orb_feat);
	//cout << "Computed " << out_orb_feat.size().height << " features!" << endl;

	Ptr<ORB> dest_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint> dest_orb_points;
	dest_orb_obj->detect(dest, dest_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat dest_out_orb_feat;
	dest_orb_obj->compute(dest, dest_orb_points, dest_out_orb_feat);

	for (int j = 0; j < feat_to_compute; j++)
	{
		if (lin)
		{
			MatchingLibs::linear_search(dest_out_orb_feat, src_out_orb_feat.row(j));
		}
		else
		{
			MatchingLibs::parallel_search(dest_out_orb_feat, branching_factor, max_leaves_amount, trees_amount,
										  max_features_to_search, top_k_feat, src_out_orb_feat.row(j));
		}
	}
}

void profile_sift(cv::Mat &src, cv::Mat &dest)
{
	Ptr<SIFT> src_sift_obj = SIFT::create(feat_to_compute);
	vector<KeyPoint> src_sift_points;
	src_sift_obj->detect(src, src_sift_points);
	//cout << "Detected " << src_sift_points.size() << " keypoints!" << endl;
	cv::Mat src_out_sift_feat;
	src_sift_obj->compute(src, src_sift_points, src_out_sift_feat);
	//cout << "Computed " << src_out_sift_feat.size().height << " features!" << endl;

	Ptr<SIFT> dest_sift_obj = SIFT::create(feat_to_compute);
	vector<KeyPoint> dest_sift_points;
	dest_sift_obj->detect(dest, dest_sift_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl;
	cv::Mat dest_out_sift_feat;
	dest_sift_obj->compute(dest, dest_sift_points, dest_out_sift_feat);

	//Rows = #features, cols = # components, type is 32F
	//std::cout << CV_MAT_TYPE(dest_out_sift_feat.type()) << std::endl;
	//Quantize the features
	cv::Mat quantized_dest_sift = MatchingLibs::median_quantize(dest_out_sift_feat);
	cv::Mat quantized_src_sift = MatchingLibs::median_quantize(src_out_sift_feat);

	for (int j = 0; j < feat_to_compute; j++)
	{
		MatchingLibs::parallel_search(quantized_dest_sift, branching_factor, max_leaves_amount, trees_amount,
									  max_features_to_search, top_k_feat, quantized_src_sift.row(j));
	}
}
