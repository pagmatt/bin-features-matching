#include "hierarchical.h"
//#include "hierarchical_lib.hh"
// Salient points and features computation
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
// Linear algebra library
// #include <armadillo>

using namespace std;
// Armadillo
// using namespace arma;
// OpenCV
using namespace cv;

int const feat_to_compute = 50;
int const max_features_to_search = 50;
int const branching_factor = 5;
int const max_leaves_amount = 5;
int const trees_amount = 2;
int const top_k_feat = 2;	// Extract just top 2, in order to use NNDR technique
int const px_to_draw = 150;
int const gap = 30;
std::string ref_path = "../testing_dataset/img_ref.png";
std::string target_path = "../testing_dataset/img2.png"; // number can be set in [1,5]

cv::Mat
find_ORB_matches (cv::Mat &src, cv::Mat &dest);

int main(int, char **)
{

	// Loading of the image files

	cv::Mat src = cv::imread(samples::findFile(ref_path), IMREAD_GRAYSCALE );
    if (src.empty())
    {
        cout << "Could not open or find the reference image!\n" << endl;
        return -1;
    }
	cv::Mat dest = cv::imread(samples::findFile(target_path), IMREAD_GRAYSCALE );
    if (dest.empty())
    {
        cout << "Could not open or find the target image!\n" << endl;
        return -1;
    }

	cv::Mat orb_matches = find_ORB_matches(src, dest);
	// Features computation : ORB comparison

	cv::imshow("ORB matches", orb_matches);
	cv::waitKey(0);

}

cv::Mat
find_ORB_matches (cv::Mat &src, cv::Mat &dest)
{
	Ptr<ORB> src_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint>src_orb_points;
	src_orb_obj->detect(src, src_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl; 
	cv::Mat src_out_orb_feat;
	src_orb_obj->compute(src, src_orb_points, src_out_orb_feat);
	//cout << "Computed " << out_orb_feat.size().height << " features!" << endl; 

	Ptr<ORB> dest_orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint>dest_orb_points;
	dest_orb_obj->detect(dest, dest_orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl; 
	cv::Mat dest_out_orb_feat;
	dest_orb_obj->compute(dest, dest_orb_points, dest_out_orb_feat);

	// Output image
	cv::Mat stacked_orb = cv::Mat::zeros(px_to_draw*feat_to_compute + gap*(feat_to_compute - 1),
										 px_to_draw*3, CV_8U); // Create stacked image canvas
	stacked_orb.setTo((cv::Scalar(255,255,255))); // Make it white

	// Search for similar features
	for(int j = 0; j < feat_to_compute; j++)	// TODO: 5 -> feat-to-compute
	{
		cv::Mat out = MatchingLibs::parallel_search(dest_out_orb_feat, branching_factor, max_leaves_amount, trees_amount, 
													max_features_to_search, top_k_feat, src_out_orb_feat.row(j));

		std::cout << out.size().height << " matches obtained!" << std::endl;
		for(int i = 0; i < out.size().height; i++)
		{
			std::cout << "Distance of the match to the query: ";
			std::cout << cv::norm(src_out_orb_feat.row(j), out.row(i), cv::NORM_HAMMING) << std::endl;
		}

		// Visualize search results: target feature
		Point2f src_kp = src_orb_points[j].pt; // Get coord of src keypoint
		cv::Rect src_to_crop(src_kp.x - px_to_draw/2, src_kp.y - px_to_draw/2, px_to_draw, px_to_draw); // x_start, y_start, width, height
		cv::Mat src_crop = src(src_to_crop);

		// Visualize search results: feature match
		// Must find the index in the original matrix, in order to associate feature to its descriptor
		Point2f dest_kp = dest_orb_points[MatchingLibs::search_feature(dest_out_orb_feat, out.row(0))].pt; // Get coord of matched, closest feature
		cv::Rect dst_to_crop(dest_kp.x - px_to_draw/2, dest_kp.y - px_to_draw/2, px_to_draw, px_to_draw); // x_start, y_start, width, height
		cv::Mat dest_crop = dest(dst_to_crop);

		//std::cout << CV_MAT_TYPE(src_crop.type()) << std::endl;
		// Stack target and match and show them
		//cv::Mat stacked_orb = cv::Mat::zeros(px_to_draw, px_to_draw*3, CV_8U); // Create stacked image canvas
		//stacked_orb.setTo((cv::Scalar(255,255,255))); // Make it white
 
		src_crop.copyTo(stacked_orb.colRange(1, px_to_draw+1).rowRange((gap+px_to_draw)*j, 
									px_to_draw + (gap+px_to_draw)*j));
		dest_crop.copyTo(stacked_orb.colRange(2*px_to_draw, 3*px_to_draw).rowRange((gap+px_to_draw)*j, 
									px_to_draw + (gap+px_to_draw)*j));
	}

	return stacked_orb;
}



	



