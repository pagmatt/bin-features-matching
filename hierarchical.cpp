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

int main(int, char **)
{
	tree<string> tr;
	tree<string>::iterator top, one, two, four, loc, banana;

	// Packages check
	// cout << "Armadillo version: " << arma_version::as_string() << endl;

	// Tree library training + printing extension
	top=tr.begin();
	one=tr.insert(top, "one");
	four=tr.insert(top, "four");
	two=tr.append_child(one, "two");
	tr.append_child(two, "apple");
	banana=tr.append_child(two, "banana");
	tr.append_child(banana,"cherry");
	tr.append_child(two, "peach");
	tr.append_child(one,"three");
	cout << "Test self-made printing" << endl;
	tr.print("", true);

	// --- Orb features computation ---

	cv::Mat src = cv::imread(samples::findFile("../Images_dataset/test.jpg"), IMREAD_GRAYSCALE );
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
	// Features computation
	Ptr<ORB> orb_obj = ORB::create(200);
	vector<KeyPoint>orb_points;
	orb_obj->detect(src, orb_points);
	cout << "Detected " << orb_points.size() << " keypoints!" << endl; 
	cv::Mat out_orb_feat;
	orb_obj->compute(src, orb_points, out_orb_feat);
	cout << "Computed " << out_orb_feat.size().height << " features!" << endl; 
	/* Show computed salient points
	cv::Mat out_image;
	drawKeypoints(src, orb_points, out_image);
	cout << "Features:" << out_orb_feat.size() << endl; 
	imshow("ORB Keypoints", out_image);
	waitKey();
	return 0;
	*/

	// Test hierarchical tree strcuture
	// cv::Mat skimmed = out_orb_feat; //(Range::all(), Range(1, 3)).clone(); // Skim features for testing purposes
	tree<cv::Mat> out_tree;
	tree<cv::Mat>::pre_order_iterator out_iter;
	out_iter = out_tree.begin();
	out_iter = out_tree.insert(out_iter, cv::Mat::zeros(out_orb_feat.size().width, 1, CV_16F)); // Dummy head
	MatchingLibs::create_search_tree(out_orb_feat, out_tree, out_iter, 5, 20);

}



	



