#include <algorithm>
#include <string>
#include <iostream>
// STL-like tree implementation
#include "tree.hh"
// OpenCV data stuctures
#include <opencv2/core/core.hpp>

class HierarchicalLibs
{
	public:
		static void create_search_tree(cv::Mat* features_set);
};