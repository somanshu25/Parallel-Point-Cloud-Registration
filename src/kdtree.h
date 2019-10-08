#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <vector>
using namespace std;

namespace KDtree {
	
	struct Node {
		Node();
		Node(glm::vec3 val, int parent);
		bool left;
		bool right;
		int parent;
		glm::vec3 value;
	};

	void createTree(vector<glm::vec3>& target,vector<KDtree::Node>& result);
	void insertNode(vector<glm::vec3>& value, vector<KDtree::Node>& result, int pos, int depth,int parent);
	//int calculateMaxDepth(vector<glm::vec3> value, int depth, int maxDepth);
}
