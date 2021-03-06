//#include <algorithm>
#include <random>
#include <string>
#include <iostream>
#include <map>
#include <bitset>
#include <algorithm>
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
	parallel_search(cv::Mat &features_set, int branch_factor, int max_leaves, int trees,
					int max_searched, int max_out, cv::Mat query);

	static cv::Mat
	linear_search(cv::Mat features_set, cv::Mat query);

	static void
	traverse_search_tree(tree<cv::Mat> &s_tree, tree<cv::Mat>::iterator from, cv::Mat &found,
						 std::vector<tree<cv::Mat>::pre_order_iterator> &refine_queue, cv::Mat query);
	static tree<cv::Mat>
	create_search_tree(cv::Mat &features_set, tree<cv::Mat> &out_tree, tree<cv::Mat>::pre_order_iterator pos,
					   int branch_factor, int max_leaves);

	static cv::Mat
	median_quantize(cv::Mat &features_set);
	// Utilities
	static int
	search_feature(cv::Mat &features_set, cv::Mat target);

private:
	static std::vector<int>
	pick_unique_rnd(int rnd_amount, int min, int max);
	static std::map<int, cv::Mat>
	partition_around_centers(std::vector<int> &centers_set, cv::Mat &features_set);
};

void MatchingLibs::merge_mat(cv::Mat &dest, cv::Mat &cand)
{
	for (int i = 0; i < cand.size().height; i++)
	{
		bool there = false;
		for (int k = 0; k < dest.size().height; k++)
		{
			// Check if such entry is already there
			cv::Mat diff = cv::Mat::zeros(1, cand.size().height, CV_8U);
			cv::bitwise_xor(dest.row(k), cand.row(i), diff);

			if (cv::countNonZero(diff) == 0)
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

	while (rnd_set.size() < rnd_amount) // Want unique rnd values
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

	for (int j = 0; j < features_set.size().height; j++)
	{
		uint32_t lucky_index;
		uint32_t dist_to_center = features_set.size().width * 32; // max hamming distance
		for (auto i = centers_set.begin(); i != centers_set.end(); i++)
		{
			auto temp_dist = cv::norm(features_set.row(*i), features_set.row(j),
									  cv::NORM_HAMMING);
			if (temp_dist < dist_to_center)
			{
				lucky_index = *i;
				dist_to_center = temp_dist;
			}
		}
		// Insert element into the map
		map_it = out_partition.find(lucky_index);
		if (map_it != out_partition.end())
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
MatchingLibs::linear_search(cv::Mat features_set, cv::Mat query)
{
	std::vector<uint16_t> distances;
	for (int i = 0; i < features_set.rows; i++)
	{
		distances.push_back(cv::norm(features_set.row(i), query, cv::NORM_HAMMING));
	}
	// Return the best match found
	cv::Mat found = cv::Mat::zeros(features_set.size().width, 1, CV_8U);
	int min_index = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
	found.row(1) = features_set.row(min_index);

	return found;
}

cv::Mat
MatchingLibs::parallel_search(cv::Mat &features_set, int branch_factor, int max_leaves, int trees,
							  int max_searched, int max_out, cv::Mat query)
{
	// Create the search trees
	std::vector<tree<cv::Mat>> tree_vec;
	for (int i = 0; i < trees; i++)
	{
		tree<cv::Mat> out_tree;
		tree<cv::Mat>::pre_order_iterator out_iter;
		out_iter = out_tree.begin();
		out_iter = out_tree.insert(out_iter, cv::Mat::zeros(features_set.size().width, 1, CV_8U)); // Dummy head
		MatchingLibs::create_search_tree(features_set, out_tree, out_iter, branch_factor, max_leaves);
		tree_vec.push_back(out_tree);
	}
	// Search
	std::vector<tree<cv::Mat>>::iterator tree_vec_it = tree_vec.begin();
	cv::Mat found;
	std::vector<tree<cv::Mat>::pre_order_iterator> refine_queue;

	while (tree_vec_it != tree_vec.end())
	{
		tree<cv::Mat>::pre_order_iterator out_iter = tree_vec_it->begin();
		MatchingLibs::traverse_search_tree(*tree_vec_it, out_iter, found, refine_queue, query);
		// Eventually refine the search
		while (found.size().height<max_searched & refine_queue.size()> 0)
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
	for (int i = 0; i < found.size().height; i++)
	{
		distances.push_back(cv::norm(found.row(i), query, cv::NORM_HAMMING));
	}
	// Create vec containing indexes of elements
	std::vector<int> sorted_idx(distances.size());
	std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
	// Sort indexes, but comparing the distances
	sort(sorted_idx.begin(), sorted_idx.end(), [&](int i, int j) { return distances[i] < distances[j]; });
	// Now, just enough to access elements pointed by first K sorted indexes
	cv::Mat out;
	for (int i = 0; i < max_out; i++)
	{
		out.push_back(found.row(sorted_idx[i]));
	}
	return out;
}

void MatchingLibs::traverse_search_tree(tree<cv::Mat> &s_tree, tree<cv::Mat>::iterator from, cv::Mat &found,
										std::vector<tree<cv::Mat>::pre_order_iterator> &refine_queue, cv::Mat query)
{

	if (s_tree.number_of_children(from) <= 1)
	{
		// Leaf node
		from = s_tree.child(from, 0);
		MatchingLibs::merge_mat(found, from.node->data);
	}
	else
	{
		// Pick non-leaf closest to query
		tree<cv::Mat>::pre_order_iterator lucky_it;
		int lucky_dist = query.size().width * 32; // Max Hamming distance
		for (int i = 0; i < s_tree.number_of_children(from); i++)
		{
			tree<cv::Mat>::pre_order_iterator lottery_it = s_tree.child(from, i);
			//std::cout << lottery_it.node->data.size();
			auto temp_dist = cv::norm(query, lottery_it.node->data, cv::NORM_HAMMING);
			if (temp_dist < lucky_dist)
			{
				lucky_dist = temp_dist;
				lucky_it = lottery_it;
			}
		}

		// Add the others to the recursion
		for (int i = 0; i < s_tree.number_of_children(from); i++)
		{
			tree<cv::Mat>::pre_order_iterator unlucky_it = s_tree.child(from, i);
			if (unlucky_it != lucky_it)
			{
				refine_queue.push_back(unlucky_it);
			}
		}

		// Recurse on the closest node
		MatchingLibs::traverse_search_tree(s_tree, lucky_it, found, refine_queue, query);
	}
}

int MatchingLibs::search_feature(cv::Mat &features_set, cv::Mat target)
{
	int found{-1};
	for (int i = 0; i < features_set.size().height; i++)
	{
		if (cv::norm(target, features_set.row(i), cv::NORM_HAMMING) == 0)
		{
			found = i;
		}
	}
	return found;
}

tree<cv::Mat>
MatchingLibs::create_search_tree(cv::Mat &features_set, tree<cv::Mat> &out_tree, tree<cv::Mat>::pre_order_iterator pos, int branch_factor, int max_leaves)
{
	int feat_amount = features_set.size().height;
	if (feat_amount < max_leaves)
	{
		// Create leaf node with all the points in the dataset
		out_tree.append_child(pos, features_set);
	}
	else
	{
		// Pick "branch_factor" random points in dataset as centers
		// and cluster around them
		tree<cv::Mat>::pre_order_iterator newPos = pos;
		std::vector<int> rnd_centers = MatchingLibs::pick_unique_rnd(branch_factor, 0, feat_amount - 1);
		std::map<int, cv::Mat> cnt_partition = MatchingLibs::partition_around_centers(rnd_centers, features_set);
		// Iterate through partition map, create nodes and recursively call the function
		std::map<int, cv::Mat>::iterator map_iter = cnt_partition.begin();
		while (map_iter != cnt_partition.end())
		{
			auto index = map_iter->first;
			newPos = out_tree.append_child(pos, features_set.row(index));

			MatchingLibs::create_search_tree(map_iter->second, out_tree, newPos, branch_factor, max_leaves);
			map_iter++;
		}
	}
	return out_tree;
}

cv::Mat
MatchingLibs::median_quantize(cv::Mat &features_set)
{
	// Assumes that the features_set OpenCV type is 5, therefore the at<float>
	// The smaller unit that cv::Mat allows is uint8, therefore 8 bits are grouped and then stored as
	// an entry of such cv::Mat
	cv::Mat quantized_feat = cv::Mat::zeros(features_set.rows, features_set.cols / 8, CV_8U); // One uint8 will store 8 bits
	for (int iFeat = 0; iFeat < features_set.rows; iFeat++)
	{
		// Find the median value of the components
		std::vector<uint8_t> feat_comp = std::vector<uint8_t>();
		for (int jComp = 0; jComp < features_set.cols; jComp++)
		{
			uint8_t temp_comp = static_cast<uint8_t>(features_set.at<float>(iFeat, jComp));
			feat_comp.push_back(temp_comp);
		}
		std::sort(feat_comp.begin(), feat_comp.end());
		uint8_t median = feat_comp.at(feat_comp.size() / 2);

		for (int jCompMaj = 0; jCompMaj < features_set.cols / 8; jCompMaj++)
		{
			std::bitset<8> temp_entry;
			for (int kCompMin = 0; kCompMin < 8; kCompMin++)
			{
				uint8_t temp_comp = static_cast<uint8_t>(features_set.at<float>(iFeat, kCompMin + jCompMaj * 8));
				temp_entry.set(kCompMin, (temp_comp > median ? 1 : 0));
			}
			// Get bitset -> cv::Mat entry
			quantized_feat.at<uchar>(iFeat, jCompMaj) = static_cast<uint8_t>(temp_entry.to_ulong());
		}
	}

	return quantized_feat;
}
