import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from hierarchy.kmeans_agglomerative import KMeansAgglomerativeClustering
from math import sqrt


class TreeNode:
    def __init__(self, idx, distance=0, count=1):
        self.idx = idx
        self.left = None
        self.right = None
        self.distance = distance
        self.count = count
        self.labels = []
        self.summarry = None  # To store AI-generated summary of labels
        
    def is_leaf(self):
        return self.left is None and self.right is None
        
        
class RecursiveHierarchicalClustering:
    def __init__(self, initial_cluster_size, run_kmeans_agglomerative_threshold):
        self.initial_cluster_size = initial_cluster_size
        self.run_kmeans_agglomerative_threshold = run_kmeans_agglomerative_threshold
        self.node_counter = 0
        
    def fit(self, X, labels=None):
        """
        Fit the recursive hierarchical clustering
        
        Args:
            X: Feature matrix
            labels: Optional labels for the data points
        """
        if labels is None:
            labels = list(range(len(X)))
            
        if self.initial_cluster_size < 2:
            self.root = self._build_tree_from_sklearn_agglomerative(
                AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=0, 
                    linkage="ward",
                    compute_distances=True
                ).fit(X), 
                labels
            )
            return self
            
        # Initial clustering
        initial_out = KMeansAgglomerativeClustering(max_n_clusters=self.initial_cluster_size).fit(X)
        
        # Group samples by initial clusters
        cluster2samples = defaultdict(list)
        cluster2encodings = defaultdict(list)
        cluster2labels = defaultdict(list)
        
        for idx, cluster_id in enumerate(initial_out.kmeans_preds_):
            cluster2samples[cluster_id].append(idx)
            cluster2encodings[cluster_id].append(X[idx])
            cluster2labels[cluster_id].append(labels[idx])
        
        # Build the tree recursively
        self.root = self._build_tree_from_agglomerative(
            initial_out, 
            cluster2encodings, 
            cluster2labels
        )
        
        return self
    
    def _build_tree_from_agglomerative(self, agglom_model, cluster2encodings, cluster2labels):
        """Build tree from agglomerative clustering results"""
        n_leaves = len(agglom_model.centers_)
        
        # Create leaf nodes for each cluster
        leaf_nodes = {}
        for i in range(n_leaves):
            leaf_nodes[i] = TreeNode(
                idx=self.node_counter,
                distance=0,
                count=len(cluster2encodings[i])
            )
            # Add labels to leaf node
            leaf_nodes[i].labels = cluster2labels[i]
            #for label in cluster2labels[i]:
            #    leaf_nodes[i].add_label(label)
            self.node_counter += 1
        
        # Build internal nodes from agglomerative clustering
        internal_nodes = {}
        
        for i, (left_idx, right_idx) in enumerate(agglom_model.children_):
            # Get left child
            if left_idx < n_leaves:
                left_child = leaf_nodes[left_idx]
            else:
                left_child = internal_nodes[left_idx - n_leaves]
            
            # Get right child  
            if right_idx < n_leaves:
                right_child = leaf_nodes[right_idx]
            else:
                right_child = internal_nodes[right_idx - n_leaves]
            
            # Create internal node
            internal_node = TreeNode(
                idx=self.node_counter,
                distance=agglom_model.distances_[i],
                count=left_child.count + right_child.count
            )
            internal_node.left = left_child
            internal_node.right = right_child
            internal_node.labels = left_child.labels + right_child.labels
            
            self.node_counter += 1
            internal_nodes[i] = internal_node
        
        # The root is the last internal node created
        root = internal_nodes[len(agglom_model.children_) - 1]
        
        # Now recursively process leaf nodes that need further clustering
        # self._recursive_cluster_leaves(root, cluster2encodings, cluster2labels)
        self._recursive_cluster_leaves(leaf_nodes, cluster2encodings, cluster2labels)
        
        print(f"Tree is built.")
        
        return root
    
    def _recursive_cluster_leaves(self, leaf_nodes, cluster2encodings, cluster2labels):
        """Recursively cluster leaf nodes that exceed the threshold"""
        for cluster_id, node in leaf_nodes.items():
            
            if len(cluster2encodings[cluster_id]) <= 1:
                continue
            
            # This leaf needs further clustering
            subX = np.vstack(cluster2encodings[cluster_id])
            sublabels = cluster2labels[cluster_id]
            
            if len(cluster2encodings[cluster_id]) > self.run_kmeans_agglomerative_threshold:
                # Use KMeansAgglomerativeClustering
                initial_cluster_size = min(self.initial_cluster_size, int(sqrt(len(cluster2encodings[cluster_id]) / 2)))
                sub_out = KMeansAgglomerativeClustering(max_n_clusters=initial_cluster_size).fit(subX)
                
                # Group sub-samples
                sub_cluster2encodings = defaultdict(list)
                sub_cluster2labels = defaultdict(list)
                
                for idx, sub_cluster_id in enumerate(sub_out.kmeans_preds_):
                    sub_cluster2encodings[sub_cluster_id].append(subX[idx])
                    sub_cluster2labels[sub_cluster_id].append(sublabels[idx])
                
                # Build subtree
                subtree_root = self._build_tree_from_agglomerative(
                    sub_out, 
                    sub_cluster2encodings, 
                    sub_cluster2labels
                )
            else:
                # Use regular AgglomerativeClustering
                sub_out = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=0, 
                    linkage="ward",
                    compute_distances=True
                ).fit(subX)
                
                subtree_root = self._build_tree_from_sklearn_agglomerative(
                    sub_out, 
                    sublabels
                )
            
            # Replace the leaf with the subtree
            node.left = subtree_root.left
            node.right = subtree_root.right
            node.distance = subtree_root.distance
    
    def _build_tree_from_sklearn_agglomerative(self, agglom_model, labels):
        """Build tree from sklearn AgglomerativeClustering results"""
        n_samples = len(labels)
        
        # Create leaf nodes
        leaf_nodes = {}
        for i in range(n_samples):
            leaf_nodes[i] = TreeNode(
                idx=self.node_counter,
                distance=0,
                count=1
            )
            leaf_nodes[i].labels = [labels[i]]
            #leaf_nodes[i].add_label(labels[i])
            self.node_counter += 1
        
        # Build internal nodes
        internal_nodes = {}
        
        for i, (left_idx, right_idx) in enumerate(agglom_model.children_):
            # Get left child
            if left_idx < n_samples:
                left_child = leaf_nodes[left_idx]
            else:
                left_child = internal_nodes[left_idx - n_samples]
            
            # Get right child
            if right_idx < n_samples:
                right_child = leaf_nodes[right_idx]
            else:
                right_child = internal_nodes[right_idx - n_samples]
            
            # Create internal node
            internal_node = TreeNode(
                idx=self.node_counter,
                distance=agglom_model.distances_[i],
                count=left_child.count + right_child.count
            )
            internal_node.left = left_child
            internal_node.right = right_child
            internal_node.labels = left_child.labels + right_child.labels
            
            self.node_counter += 1
            internal_nodes[i] = internal_node
            
        print(f"Tree is built.")
        
        # Return the root (last internal node)
        return internal_nodes[len(agglom_model.children_) - 1]