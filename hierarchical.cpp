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
	cout << "Detected keypoints:" << orb_points.size() << endl; 
	cv::Mat out_orb_feat;
	orb_obj->compute(src, orb_points, out_orb_feat);
	/* Show computed salient points
	cv::Mat out_image;
	drawKeypoints(src, orb_points, out_image);
	cout << "Features:" << out_orb_feat.size() << endl; 
	imshow("ORB Keypoints", out_image);
	waitKey();
	return 0;
	*/

	// Test hierarchical tree strcuture
	HierarchicalLibs::create_search_tree(out_orb_feat, 5, 10);
}

tree<cv::Mat> 
HierarchicalLibs::create_search_tree(cv::Mat features_set, int branch_factor, int max_leaves)
{
	/* Inform user of tree creation
    cout << "Creating hierarchical search structure, for ";
	cout << features_set.size().height << " features" <<  endl;
	*/
	tree<cv::Mat> out_tree;
	tree<cv::Mat>::iterator top;
	int feat_amount = features_set.size().height;

	top = out_tree.begin();

	if(feat_amount << max_leaves)
	{
		// Create leaf node with all the points in the dataset
		out_tree.insert(top, features_set);
	}
	else
	{
		std::vector<u_int16_t> rnd_centers;

		// Pick "branch_factor" random points in dataset as centers
		// and cluster around them
		

	}
	return out_tree;
}
void 
HierarchicalLibs::pick_unique_rnd(vector<u_int16_t> &rnd_unique_set, int min, int max)
{
	std::default_random_engine rng_engine;
	std::uniform_int_distribution<int> distribution(min, max);

	// distribution(rng_engine)

}
	



