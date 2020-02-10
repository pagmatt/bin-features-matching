//#include <algorithm>
#include <random>
#include <string>
#include <iostream>
// STL-like tree implementation
#include "tree.hh"
#include <map> 
// OpenCV data stuctures
#include <opencv2/core/core.hpp>

class MatchingLibs
{
	public:
		static tree<cv::Mat> 
		create_search_tree(cv::Mat features_set, tree<cv::Mat> &out_tree, tree<cv::Mat>::pre_order_iterator pos, int branch_factor, int max_leaves);
	private:
		static std::vector<int>
		pick_unique_rnd(int rnd_amount, int min, int max);
		static std::map<int, cv::Mat>
		partition_around_centers(std::vector<int> &centers_set, cv::Mat &features_set);
};

std::vector<int>
MatchingLibs::pick_unique_rnd(int rnd_amount, int min, int max)
{
	std::random_device r;
	std::set<int> rnd_set = std::set<int>();
	std::default_random_engine rng_engine{r()};
	std::uniform_int_distribution<int> distribution(min, max);

	while(rnd_set.size() < rnd_amount)	// Want unique rnd values
	{
		rnd_set.insert(distribution(rng_engine));
	}
	std::vector<int> out(rnd_set.begin(), rnd_set.end());
	return out;
}

std::map<int, cv::Mat>
MatchingLibs::partition_around_centers(std::vector<int> &centers_set, cv::Mat &features_set)
{
	std::map<int, cv::Mat> out_partition;
	tree<cv::Mat> out_tree;
	std::map<int, cv::Mat>::iterator map_it;
	//std::cout << centers_set.size() << " centers";
	for(int j=0; j<features_set.size().height; j++)
	{	
		uint32_t lucky_index;
		uint32_t dist_to_center = features_set.size().width*32; // max hamming distance
		for (auto i = centers_set.begin(); i != centers_set.end(); i++)
		{
			//std::cout << "Trying " << *i << std::endl; 
			//std::cout << "Features size " << features_set.size() << std::endl; 
			auto temp_dist = cv::norm(features_set.row(*i), features_set.row(j), 
										cv::NORM_HAMMING); 
			if(temp_dist < dist_to_center)
			{
				lucky_index = *i;
				dist_to_center = temp_dist;
			}
		}
		// Insert element into the map
		map_it = out_partition.find(lucky_index);
		if(map_it != out_partition.end())
		{
			// Center already present, stack to its feature set
			out_partition[lucky_index].push_back(features_set.row(j));
		}
		else
		{
			out_partition[lucky_index] = features_set.row(j);
		}
		//std::cout << lucky_index << " @ distance: " << dist_to_center << std::endl;
	}

	return out_partition;
}

tree<cv::Mat> 
MatchingLibs::create_search_tree(cv::Mat features_set, tree<cv::Mat> &out_tree, tree<cv::Mat>::pre_order_iterator pos, int branch_factor, int max_leaves)
{
	/* Inform user of tree creation
    cout << "Creating hierarchical search structure, for ";
	cout << features_set.size().height << " features" <<  endl;
	*/
	int feat_amount = features_set.size().height;

	if(feat_amount < max_leaves)
	{
		// Create leaf node with all the points in the dataset
		out_tree.append_child(pos, features_set);
	}
	else
	{	
		// Pick "branch_factor" random points in dataset as centers
		// and cluster around them
		tree<cv::Mat>::pre_order_iterator newPos = pos;
		std::vector<int> rnd_centers = MatchingLibs::pick_unique_rnd(branch_factor, 0, feat_amount-1);
		std::map<int, cv::Mat> cnt_partition = MatchingLibs::partition_around_centers(rnd_centers, features_set);
		// Iterate through partition map, create nodes and recursively call the function
		std::map<int, cv::Mat>::iterator map_iter = cnt_partition.begin();
		while(map_iter != cnt_partition.end())
		{
			auto index = map_iter->first;
			newPos = out_tree.append_child(pos, features_set.row(index));
			
			MatchingLibs::create_search_tree(map_iter->second, out_tree, newPos, branch_factor, max_leaves);
			map_iter++;
		}
	}
	return out_tree;
}
