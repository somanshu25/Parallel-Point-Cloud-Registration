#pragma once

#include <glm/vec3.hpp>

namespace KDtree {
	
	struct Node {
		Node();
		Node(glm::vec3 val, int parent);
		bool left;
		bool right;
		int parent;
		glm::vec3 val;
	};

	void createTree(vector<glm::vec3>& target, glm::vec3 *result);
	void insertNode(vector<glm::vec3> value, glm::vec3 *result, int pos, int depth,int parent);
	int calculateMaxDepth(vector<glm::vec3> value, int depth, int maxDepth);
}
