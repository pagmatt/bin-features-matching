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

int const feat_to_compute = 100;
int const max_features_to_search = 50;
int const branching_factor = 5;
int const max_leaves_amount = 5;
int const trees_amount = 2;
int const top_k_feat = 5;
std::string ref_path = "../testing_dataset/img_ref.png";

int main(int, char **)
{
	// --- Orb features computation ---

	cv::Mat src = cv::imread(samples::findFile(ref_path), IMREAD_GRAYSCALE );
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
	// Features computation

	Ptr<ORB> orb_obj = ORB::create(feat_to_compute);
	vector<KeyPoint>orb_points;
	orb_obj->detect(src, orb_points);
	//cout << "Detected " << orb_points.size() << " keypoints!" << endl; 
	cv::Mat out_orb_feat;
	orb_obj->compute(src, orb_points, out_orb_feat);
	//cout << "Computed " << out_orb_feat.size().height << " features!" << endl; 

	// Test hierarchical tree strcuture
	// cv::Mat skimmed = out_orb_feat; //(Range::all(), Range(1, 3)).clone(); // Skim features for testing purposes
	// Test search
	cv::Mat out = MatchingLibs::parallel_search(out_orb_feat, branching_factor, max_leaves_amount,
													 trees_amount, max_features_to_search, top_k_feat, out_orb_feat.row(57));
	std::cout << out.size().height << " matches obtained!" << std::endl;
	for(int i = 0; i < out.size().height; i++)
	{
		std::cout << "Distance of the match to the query: ";
		std::cout << cv::norm(out_orb_feat.row(57), out.row(i), cv::NORM_HAMMING) << std::endl;
	}
}



	



