#include <algorithm>
#include <random>
#include <string>
#include <iostream>
// STL-like tree implementation
#include "tree.hh"
// OpenCV data stuctures
#include <opencv2/core/core.hpp>

class HierarchicalLibs
{
	public:
		static tree<cv::Mat> 
		create_search_tree(cv::Mat features_set, int branch_factor, int max_leaves);
	private:
		void pick_unique_rnd(vector<u_int16_t> &rnd_unique_set, int min, int max);
};