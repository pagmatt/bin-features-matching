#include <algorithm>
#include <random>
#include <string>
#include <iostream>
// STL-like tree implementation
#include "tree.hh"
// OpenCV data stuctures
#include <opencv2/core/core.hpp>

class MatchingLibs
{
	public:
		static tree<cv::Mat> 
		create_search_tree(cv::Mat features_set, int branch_factor, int max_leaves);
	private:
		static void 
		pick_unique_rnd(std::set<u_int16_t> &rnd_set, int rnd_amount, int min, int max);

		static tree<cv::Mat> 
		partition_around_centers(std::set<u_int16_t> centers_set, cv::Mat features_set);
};