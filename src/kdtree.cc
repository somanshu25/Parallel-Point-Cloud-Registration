#include <vector>
#include<stdio.h>
#include "kdtree.h"
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

using namespace std;

bool sortbyX(const glm::vec3 &a, const glm::vec3 &b) {
	return a.x < b.x;
}

bool sortbyY(const glm::vec3 &a, const glm::vec3 &b) {
	return a.y < b.y;
}

bool sortbyZ(const glm::vec3 &a, const glm::vec3 &b) {
	return a.z < b.z;
}

void KDtree::createTree(vector<glm::vec3>& target,vector<KDtree::Node> result) {
	insertNode(target,result,0,0,-1);
}

void KDTree::insertNode(vector<glm::vec3>& value, vector<KDtree::Node>& result, int pos, int depth, int parent) {
	if (depth % 3 == 0) {
		// Sort by X
		sort(value.begin(), value.end(), sortbyX);
	}
	else if (depth % 3 == 1) {
		// Sort by Y
		sort(value.begin(), value.end(), sortbyY);
	}
	else {
		// Sort by Z
		sort(value.begin(), value.end(), sortbyZ);
	}

	int mid = value.size() / 2;
	result[pos] = KDtree::Node( value[mid],parent);

	// Insert into the left child elements
	if (mid > 0) {
		result[pos].left = true;
		insertNode(vector<glm::vec3> left(value.begin(), value.begin() + mid), result, (2 * pos) + 1, depth++, pos);
	}

	// Insert into the right child elements
	if (mid < value.size() - 1) {
		result[pos].right = true;
		insertNode(vector<glm::vec3> right(value.begin() + mid + 1, value.end()), result, (2 * pos) + 2, depth++, pos);
	}

}
/*
int KDTre::calculateMaxDepth(vector<glm::vec3> value,int depth,int maxDepth) {
	if (depth % 3 == 0) {
		// Sort by X
		sort(value.begin(), value.end(), sortbyX);
	}
	else if (*depth % 3 == 1) {
		// Sort by Y
		sort(value.begin(), value.end(), sortbyY);
	}
	else {
		// Sort by Z
		sort(value.begin(), value.end(), sortbyZ);
	}

	int mid = value.size() / 2;

	// Insert into the left child elements
	if (mid > 0) {
		maxDepth = insertNode(vector<glm::vec3> left(value.begin(), value.begin() + mid), depth++, maxDepth);
	}
		
	// Insert into the right child elements
	if (mid < value.size() - 1) {
		maxDepth = insertNode(vector<glm::vec3> right(value.begin() + mid + 1, value.end()), depth++,maxDepth);
	}
	if (depth > maxDepth)
		return depth;
}
*/

KDtree::Node::Node(glm::vec3 val,int p) {
	left = false;
	right = false;
	parent = p;
	value = val;
}