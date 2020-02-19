//#include <algorithm>
#include <random>
#include <string>
#include <iostream>
#include <map> 
// STL-like tree implementation
#include "tree.hh"
// OpenCV data stuctures
#include <opencv2/core/core.hpp>


class MatchingLibs
{
	public:
		static void
		merge_mat(cv::Mat &dest, cv::Mat &cand);
		static cv::Mat
		parallel_search(cv::Mat features_set, int branch_factor, int max_leaves, int trees, 
							int max_searched, int max_out, cv::Mat query);
		static void
		traverse_search_tree(tree<cv::Mat> &s_tree, tree<cv::Mat>::iterator from, cv::Mat &found, 
								std::vector<tree<cv::Mat>::pre_order_iterator> &refine_queue, cv::Mat query);
		static tree<cv::Mat> 
		create_search_tree(cv::Mat features_set, tree<cv::Mat> &out_tree, tree<cv::Mat>::pre_order_iterator pos, 
								int branch_factor, int max_leaves);
	private:
		static std::vector<int>
		pick_unique_rnd(int rnd_amount, int min, int max);
		static std::map<int, cv::Mat>
		partition_around_centers(std::vector<int> &centers_set, cv::Mat &features_set);

};

void
MatchingLibs::merge_mat(cv::Mat &dest, cv::Mat &cand)
{
	for(int i = 0; i < cand.size().height; i++)
	{
		bool there = false;
		for(int k = 0; k < dest.size().height; k++)
		{
			if(cv::norm(dest.row(k), cand.row(i), cv::NORM_HAMMING) == 0)
			{
				there = true;
				break;
			}
		}
		if (!there)
		{
			dest.push_back(cand.row(i));
		}
	}
}

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


cv::Mat
MatchingLibs::parallel_search(cv::Mat features_set, int branch_factor, int max_leaves, int trees, 
								int max_searched, int max_out, cv::Mat query)
{
	// Create the search trees
	std::vector<tree<cv::Mat>> tree_vec;
	for(int i=0; i < trees; i++)
	{
		tree<cv::Mat> out_tree;
		tree<cv::Mat>::pre_order_iterator out_iter;
		out_iter = out_tree.begin();
		out_iter = out_tree.insert(out_iter, cv::Mat::zeros(features_set.size().width, 1, CV_16F)); // Dummy head
		MatchingLibs::create_search_tree(features_set, out_tree, out_iter, 5, 50);
		tree_vec.push_back(out_tree);
	}
	// Search
	std::vector<tree<cv::Mat>>::iterator tree_vec_it = tree_vec.begin();
	cv::Mat found;
	std::vector<tree<cv::Mat>::pre_order_iterator> refine_queue;

	while(tree_vec_it != tree_vec.end())
	{
		tree<cv::Mat>::pre_order_iterator out_iter = tree_vec_it->begin();
		MatchingLibs::traverse_search_tree(*tree_vec_it, out_iter, found, refine_queue, query);
		// Eventually refine the search
		while(found.size().height < max_searched & refine_queue.size() > 0)
		{
			// Extract the closest unsearched node from the queue and recurse from there 
			tree<cv::Mat>::pre_order_iterator refine_node = refine_queue.back();
			refine_queue.pop_back();
			MatchingLibs::traverse_search_tree(*tree_vec_it, refine_node, found, refine_queue, query);
		}
		tree_vec_it++;
	}
	// Return top K closest features to query
	std::vector<int> distances;
	for(int i = 0; i < found.size().height; i++)
	{
		distances.push_back(cv::norm(found.row(i), query, cv::NORM_HAMMING));
	}
	// Create vec containing indexes of elements
	std::cout << found.size().height << " possible, unskimmed matches found!" << std::endl;
	std::vector<int> sorted_idx(distances.size());
	std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
	// Sort indexes, but comparing the distances
	sort(sorted_idx.begin(), sorted_idx.end(), [&](int i,int j) {return distances[i]<distances[j];});
	// Now, just enough to access elements pointed by first K sorted indexes
	cv::Mat out;
	for(int i = 0; i < max_out; i++)
	{	
		out.push_back(found.row(sorted_idx[i]));
	}
	return out;
}



void
MatchingLibs::traverse_search_tree(tree<cv::Mat> &s_tree, tree<cv::Mat>::iterator from, cv::Mat &found, 
									std::vector<tree<cv::Mat>::pre_order_iterator> &refine_queue, cv::Mat query)
{

	if(s_tree.number_of_children(from) <= 1)
	{
		// Leaf node
		from = s_tree.child(from, 0);
		MatchingLibs::merge_mat(found, from.node->data);
	}
	else
	{
		// Pick non-leaf closest to query
		tree<cv::Mat>::pre_order_iterator lucky_it;
		int lucky_dist = query.size().width*32; // Max Hamming distance
		for(int i=0; i < s_tree.number_of_children(from); i++)
		{
			tree<cv::Mat>::pre_order_iterator lottery_it = s_tree.child(from, i);
			//std::cout << lottery_it.node->data.size();
			auto temp_dist = cv::norm(query, lottery_it.node->data, cv::NORM_HAMMING); 
			if(temp_dist < lucky_dist)
			{
				lucky_dist = temp_dist;
				lucky_it = lottery_it;
			}
		}
		// Recurse on such node
		MatchingLibs::traverse_search_tree(s_tree, lucky_it, found, refine_queue, query);
		// Add the others to the recursion 
		for(int i=0; i < s_tree.number_of_children(from); i++)
		{
			tree<cv::Mat>::pre_order_iterator unlucky_it = s_tree.child(from, i);
			if(unlucky_it != lucky_it)
			{
				refine_queue.push_back(unlucky_it);
			}
		}

	}
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
