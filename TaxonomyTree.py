from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from promptLLM import open_route_ai_models
import json
import pathlib

parent_dir = pathlib.Path(__file__).resolve().parent

path = parent_dir / "prompts" / "leaves-summarizer-final.txt"
with open(path, 'r') as file:
    prompt_leaves_txt = file.read()

path = parent_dir / "prompts" / "internal-nodes-summarizer-final.txt"
with open(path, 'r') as file:
    prompt_internal_nodes_txt = file.read()


@dataclass
class TaxonNode:
    count: int
    distance: float
    labels: List[str] = field(default_factory=list)
    children: List["TaxonNode"] = field(default_factory=list)
    summarry: str = None  # To store AI-generated summary of labels
    parent: Optional["TaxonNode"] = None  # <-- Add this
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["labels"]
        return state
    
    def __setstate__(self, state):
        state["labels"] = []  # restore as empty list on load
        self.__dict__.update(state)

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def summarize_labels_AI(self, summarizer_model="qwen/qwen3-235b-a22b-2507"):
        if self.summarry is not None:
            return 
        
        if not self.labels:
            return "No labels"
        
        # get sibling summaries if available
        sibling_summaries = []
        if self.parent:
            for sibling in self.parent.children:
                if sibling is not self and sibling.summarry:
                    sibling_summaries.append(sibling.summarry)
        sibling_summaries_str = "; ".join(sibling_summaries) if sibling_summaries else "none"
        
        
        # get parent path if available
        parent_path = []
        while self.parent:
            if self.parent.summarry:
                # convert the root to Privacy/Securrity
                if self.parent.summarry.lower() == "root":
                    parent_path.append("Privacy/Security")
                else:
                    parent_path.append(self.parent.summarry)
            self.parent = self.parent.parent
        parent_path_str = " > ".join(reversed(parent_path)) if parent_path else "Privacy/Security"
        

        labels = "; ".join(self.labels)
        
        prompt = prompt_leaves_txt.replace("[$LABELS$]", labels)
        
        prompt = prompt.replace("[$SIBLINGS$]", sibling_summaries_str)
        
        prompt = prompt.replace("[$PARENT_PATH$]", parent_path_str)

        for attempt in range(3):
            response = open_route_ai_models(
                prompt=prompt,
                model_name=summarizer_model,
                max_tokens=50,
                json_output=True
            )
            json_part = response[response.find("{"): response.rfind("}") + 1].strip().lower()
            try:
                self.summarry = json.loads(json_part)["summary"]
                break
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"Warning: attempt {attempt + 1} failed to parse summary, retrying...")
        else:
            print(f"Error: all retries failed to parse summary for labels {self.labels}")

        print(f"{parent_path_str} || {sibling_summaries_str} is summarized as: {self.summarry}")

        return

    def summarize_children_summaries(self, summarizer_model="qwen/qwen3-235b-a22b-2507"):
        if self.summarry is not None:
            return 
        
        children_summaries = "; ".join(
            [child.summarry for child in self.children if child.summarry]
        )
        
        sibling_summaries = []
        if self.parent:
            for sibling in self.parent.children:
                if sibling is not self and sibling.summarry:
                    sibling_summaries.append(sibling.summarry)
        sibling_summaries_str = "; ".join(sibling_summaries) if sibling_summaries else "none"
        
        
        # get parent path if available
        parent_path = []
        while self.parent:
            if self.parent.summarry:
                # convert the root to Privacy/Securrity
                if self.parent.summarry.lower() == "root":
                    parent_path.append("Privacy/Security")
                else:
                    parent_path.append(self.parent.summarry)
            self.parent = self.parent.parent
        parent_path_str = " > ".join(reversed(parent_path)) if parent_path else "none"
            
        prompt = prompt_internal_nodes_txt.replace("[$LABELS$]", children_summaries)
        
        prompt = prompt.replace("[$SIBLINGS$]", sibling_summaries_str)
        
        prompt = prompt.replace("[$PARENT_PATH$]", parent_path_str)

        for attempt in range(3):
            response = open_route_ai_models(
                prompt=prompt,
                model_name=summarizer_model,
                max_tokens=50,
                json_output=True
            )
            json_part = response[response.find("{"): response.rfind("}") + 1].strip().lower()
            try:
                self.summarry = json.loads(json_part)["parent_label"]
                break
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"Warning: attempt {attempt + 1} failed to parse parent_label, retrying...")
        else:
            print(f"Error: all retries failed to parse parent_label for children {children_summaries}")

        print(f"{parent_path_str} || {sibling_summaries_str} || {children_summaries} is summarized as: {self.summarry}")

        return
    

def convert_binary_to_multiway(root, leaves, tol = 1e-6) -> TaxonNode:
    """
    Convert a binary TreeNode dendrogram into a multiway TaxonNode tree by collapsing
    any child whose distance is within `tol` of its parent's distance.
    
    - `distance` is treated as the dendrogram height at a node.
    - `count` of internal nodes is computed as the sum of children's counts.
    - `labels` of internal nodes are aggregated from the full subtree.
    - Leaf nodes keep their own `distance`, `count`, and `labels` as-is.
    """

    # --- helpers -------------------------------------------------------------
    
    leaf_idx_set: Set[int] = {lf.idx for lf in leaves}
    leaf_list: List[TaxonNode] = []   # accumulator

    def expand_relative_to_parent(child, parent_height):
        """
        If child's distance is within tol of parent height and child is internal,
        collapse it: lift its children up (recursively, so plateaus can fan out).
        Otherwise keep the child.
        """
        if (child.idx not in leaf_idx_set) and abs(child.distance - parent_height) <= tol:
            lifted = []
            if child.left:
                lifted.extend(expand_relative_to_parent(child.left, parent_height))
            if child.right:
                lifted.extend(expand_relative_to_parent(child.right, parent_height))
            return lifted
        else:
            return [child]

    # --- core recursive build -----------------------------------------------

    def build_multi(node, parent) -> TaxonNode:
        if node.idx in leaf_idx_set:
            tleaf = TaxonNode(
                count=node.count,
                distance=node.distance,
                labels=list(node.labels) if node.labels else [],
                children=[],
                parent=parent
            )
            leaf_list.append(tleaf)
            return tleaf

        parent_h = node.distance

        # Collapse near-coheight children into a single multiway parent
        raw_children = []
        if node.left:
            raw_children.extend(expand_relative_to_parent(node.left, parent_h))
        if node.right:
            raw_children.extend(expand_relative_to_parent(node.right, parent_h))
            
        # convert node to TaxonNode
        tnode = TaxonNode(
            count=node.count,
            distance=node.distance,
            labels=list(node.labels),
            children=[], # will fill in next
            parent=parent
        )

        # Recursively convert each kept/lifted child
        kids: List[TaxonNode] = [build_multi(c, tnode) for c in raw_children]

        # Assign children to the current node
        tnode.children = kids

        return tnode

    return build_multi(root, None), leaf_list
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


def binary_to_multiway_taxonomy(
    root,
    leaves,
    node_id_to_row: Dict[int, int],
    coef,                       # np.ndarray from inconsistent(Z)[:,3]
    threshold: float = 1.2,     # higher → fewer contractions (stricter)
) -> TaxonNode:
    """
    Convert a binary TreeNode dendrogram to a multi-way taxonomy by
    *contracting* merges whose inconsistency coefficient is below `threshold`.

    - Each weak merge (coef[row] < threshold) is replaced by its children
      (i.e., we 'promote' grandchildren), creating non-binary nodes.
    - Leaves keep their labels as-is.

    Parameters
    ----------
    root : TreeNode
        Root of your binary tree.
    node_id_to_row : dict(int -> int)
        Row indices from your earlier `tree_to_linkage` call; only defined for internal nodes.
    coef : np.ndarray
        Inconsistency coefficients, i.e., `inconsistent(Z, d=... )[:,3]`, aligned to rows of Z.
    threshold : float
        Weak/strong cutoff. Typical range ~1.0–1.5.

    Returns
    -------
    TaxonNode
        Root of a multi-way taxonomy tree with `.children` lists.
    """
    
    leaf_idx_set: Set[int] = {lf.idx for lf in leaves}
    leaf_list: List[TaxonNode] = []   # accumulator

    def convert(node) -> TaxonNode:
        if node.idx in leaf_idx_set:
            tleaf = TaxonNode(
                count=node.count,
                distance=node.distance,
                labels=list(node.labels),
                children=[]
            )
            leaf_list.append(tleaf)
            return tleaf
        """
        if node.is_leaf():
            return TaxonNode(
                count=node.count,
                distance=node.distance,
                labels=list(node.labels),
                children=[]
            )
        """

        left = convert(node.left)
        right = convert(node.right)

        # Build this node in the taxonomy view
        tnode = TaxonNode(
            count=node.count,
            distance=node.distance,
            labels=(left.labels + right.labels),
            children=[left, right]
        )

        # Decide whether to contract this merge
        row = node_id_to_row.get(node.idx, None)   # None for leaves
        is_weak = False
        if row is not None:
            c = float(coef[row])
            # weak if coef below threshold
            is_weak = (c < threshold)

        if is_weak:
            # Promote grandchildren (flatten one level)
            flattened = []
            for ch in tnode.children:
                if ch.children:
                    flattened.extend(ch.children)
                else:
                    flattened.append(ch)
            tnode.children = flattened

        return tnode

    # One pass contracts at each weak node. Because we decide bottom-up,
    # stacked weak links collapse naturally into multi-child nodes.
    root = convert(root)
    return root, leaf_list


def paths_to_taxon_tree(paths: list) -> "TaxonNode":
    """Build a TaxonNode tree from a list of root-to-leaf path strings.
    Each path is a ' -> ' separated string, e.g. 'Animals -> Mammals -> Dog'.
    Nodes with the same name at the same tree position are merged."""
    root = TaxonNode(count=0, distance=float("inf"), labels=[], summarry="root", parent=None)

    for path_str in paths:
        parts = [p.strip() for p in path_str.split(" -> ")]
        current = root
        for name in parts:
            match = next((c for c in current.children if c.summarry == name), None)
            if match is None:
                match = TaxonNode(count=0, distance=0.0, labels=[], summarry=name, parent=current)
                current.children.append(match)
            current = match

    # If the taxonomy already has a single real root, promote it instead of
    # keeping the synthetic wrapper node.
    if len(root.children) == 1:
        real_root = root.children[0]
        real_root.parent = None
        return real_root

    return root
