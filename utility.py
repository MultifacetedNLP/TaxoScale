import numpy as np
import textwrap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import textwrap
from typing import Tuple, Dict, Optional
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def add_new_child_labels(root, leaf_nodes, new_label_embeddings):
    # Step 1: Build id2node mapping
    id2node = build_id2node(root)

    # Step 2: For each new label, find the best matching leaf node
    for i, new_emb in enumerate(new_label_embeddings):
        best_sim = -1.0
        best_node = None
        
        for leaf in leaf_nodes:
            # Assuming leaf has an attribute 'embedding' which is L2-normalized
            sim = np.dot(new_emb, leaf.embedding)
            if sim > best_sim:
                best_sim = sim
                best_node = leaf
        
        # Step 3: Add new label to the best matching leaf node
        if best_node is not None:
            if not hasattr(best_node, 'labels'):
                best_node.labels = []
            best_node.labels.append(f"New Label {i+1}")  # or use actual label text if available
    

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    pattern = r'^```\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', cleaned_string, flags=re.DOTALL)
    return cleaned_string.strip()

def add_summary_leaves(leaf_nodes, summarizer_model="qwen/qwen3-235b-a22b-2507"):
    for node in tqdm(leaf_nodes, desc="Summarizing leaf nodes"):
        node.summarize_labels_AI(summarizer_model)



def add_summary(root, leaf_nodes, summarizer_model="qwen/qwen3-235b-a22b-2507"):
    add_summary_leaves(leaf_nodes, summarizer_model)
    
    all_nodes = []

    def collect_nodes_post(node):
        for child in node.children:
            collect_nodes_post(child)

        # append AFTER visiting children -> deepest to root
        if node not in leaf_nodes:
            all_nodes.append(node)

    collect_nodes_post(root)

    # Step 2: Process nodes with tqdm progress bar
    for node in tqdm(all_nodes, desc="Summarizing other nodes"):
        node.summarize_children_summaries(summarizer_model)



def get_paths(node: dict, current_path: list = None) -> list:
    if current_path is None:
        current_path = []
    new_path = current_path + [node.summarry] if node.summarry is not None else current_path
    if not node.children:
        return [" -> ".join(new_path)]
    paths = []
    for child in node.children:
        paths.extend(get_paths(child, new_path))
    return paths


def get_paths_from_text(taxonomy_text: str) -> list:
    lines = [line.strip() for line in taxonomy_text.strip().split('\n') if line.strip()]
    
    nodes = {}  # number_str -> label
    children = {}  # number_str -> list of child number_strs
    roots = []
    
    for line in lines:
        if line == 'The current taxonomy is:':
            continue
        
        parts = line.split(' ', 1)
        number = parts[0]
        label = parts[1] if len(parts) > 1 else ''
        
        nodes[number.rstrip(".")] = label
        children[number.rstrip(".")] = []
        
        # Find parent by removing last segment
        segments = number.split('.')
        if not segments[1]:
            roots.append(number.rstrip("."))
        else:
            parent = '.'.join(segments[:-1])
            if parent in children:
                children[parent].append(number)
    
    def build_paths(number, current_path):
        new_path = current_path + [nodes[number]]
        if not children[number]:
            return [" -> ".join(new_path)]
        paths = []
        for child in children[number]:
            paths.extend(build_paths(child, new_path))
        return paths
    
    all_paths = []
    for root in roots:
        all_paths.extend(build_paths(root, []))
    return all_paths


def get_levels(node) -> list:
    """
    Recursively traverse the tree and collect a list of dictionaries, each containing:
      - 'parent': the parent node’s aspect_name
      - 'siblings': a list of aspect_names of all children of that parent.
    """
    result = []
    if len(node.children) > 0:
        parent_name = node.summarry
        siblings = [child.summarry for child in node.children if child.summarry]
        result.append({"parent": parent_name, "siblings": siblings})
        for child in node.children:
            result.extend(get_levels(child))
    return result



def get_levels_from_text(taxonomy_text: str) -> list:
    lines = [line.strip() for line in taxonomy_text.strip().split('\n') if line.strip()]
    
    nodes = {}  # number_str -> label
    children = {}  # number_str -> list of child number_strs
    roots = []
    
    for line in lines:
        if line == 'The current taxonomy is:':
            continue
        
        parts = line.split(' ', 1)
        number = parts[0]
        label = parts[1] if len(parts) > 1 else ''
        
        nodes[number.rstrip(".")] = label
        children[number.rstrip(".")] = []
        
        # Find parent by removing last segment
        segments = number.split('.')
        if not segments[1]:
            roots.append(number.rstrip("."))
        else:
            parent = '.'.join(segments[:-1])
            if parent in children:
                children[parent].append(number)
    
    def build_levels(number):
        result = []
        if children[number]:
            parent_name = nodes[number]
            siblings = [nodes[child] for child in children[number] if nodes[child]]
            result.append({"parent": parent_name, "siblings": siblings})
            for child in children[number]:
                result.extend(build_levels(child))
        return result
    
    all_levels = []
    for root in roots:
        all_levels.extend(build_levels(root))
    return all_levels




def build_message_paper_task_labels(title, abstract):
    out = f"""
<input>
<type_definition>
Task: Defines and categorizes research efforts aimed at solving specific problems or objectives within a given field, such as classification, prediction, or optimization.
</type_definition>
<paper_title>
{title}
</paper_title>
<paper_abstract>
{abstract}
</paper_abstract>
</input>

Given the paper title and abstract above, identify the primary task that the research addresses.

Requirements for the task label:
- Start with the core task type (e.g., classification, generation, question answering, segmentation)
- Add specific modifiers that distinguish this task from related variants
- Use domain-specific terminology when applicable
- Avoid generic prefixes like "the task of", "performing", or "doing"
- Avoid generic terms like "ai", "ml", "system", "model", "framework", "llm"
- Keep it concise (2-5 words typically, each word should be informative)
- Focus on the task being solved, not the application domain or contribution

Good examples:
- "structured table question answering" (base: question answering + modifier: structured tables)
- "few shot image classification" (base: classification + modifier: few-shot and image)
- "neural machine translation" (base: translation + modifier: neural)
- "semantic image segmentation" (base: segmentation + modifier: semantic and images)

Poor examples:
- "classification" (too generic, needs modifiers)
- "the task of classifying images" (contains uninformative tokens like "the task of")
- "low resource nlp" (describes setting/contribution, not the task)
- "AI system for tables" (vague, not a specific task)

Output format:
{{
    "new_subtopic_label": "<concise_task_label>"
}}
"""
    return out


#def build_message_paper_task_labels(title, abstract):
#   out = f"""
"""<input>
<type_definition>
Task: Defines and categorizes research efforts aimed at solving specific problems or objectives within a given field, such as classification, prediction, or optimization.
</type_definition>

<paper_title>
{title}
</paper_title>

<paper_abstract>
{abstract}
</paper_abstract>

</input>

Given the input paper title and abstract, identify its task class label. In other words, answer the question: what type of task does the paper propose?

Your output should be in the following JSON format:
{{
"new_subtopic_label": <value type is string; string is a new topic label (a type of task)>,
}}
"""
#   return out


def build_id2node(root):
    """Return dict: node_id -> ClusterNode (covers leaves & internal)"""
    stack = [root]
    id2node = {}
    while stack:
        nd = stack.pop()
        id2node[nd.id] = nd
        if not nd.is_leaf():
            stack.append(nd.left)
            stack.append(nd.right)
    return id2node

def leaf_ids_under(node):
    """Return list of original leaf indices under `node` (0..n-1)."""
    if node.is_leaf():
        # for leaves, node.id is the original index into labels
        return [node.id]
    return leaf_ids_under(node.left) + leaf_ids_under(node.right)


def get_new_leaves(node, min_size=10):
    # Base case: pure leaf
    if node.is_leaf():
        return [node]
    
    # If BOTH children are small, replace them with the parent
    if node.left and node.right and node.left.count < min_size and node.right.count < min_size:
        return [node]

    leaves = []

    # Left child
    if node.left:
        if node.left.count >= min_size:
            leaves.extend(get_new_leaves(node.left, min_size))
        else:
            leaves.append(node.left)

    # Right child
    if node.right:
        if node.right.count >= min_size:
            leaves.extend(get_new_leaves(node.right, min_size))
        else:
            leaves.append(node.right)

    return leaves


def choose_k_by_silhouette(X_np, coarse_grid, window):
    """
    Two-stage (coarse -> fine) selection of k using cosine silhouette.

    Returns:
        best_k, best_model, best_score, best_labels, coarse_scores, fine_scores
    """
    N = len(X_np)
    if N < 2:
        raise ValueError("Need at least 2 samples.")

    def fit_and_score(k: int):
        if k < 2 or k >= N:
            return None, None, None
        model = KMeans(n_clusters=k, algorithm="elkan", random_state=42)
        labels = model.fit_predict(X_np)
        # silhouette needs at least 2 non-empty clusters
        if len(set(labels)) < 2:
            return None, None, None
        sil = silhouette_score(X_np, labels, random_state=42)
        return sil, model, labels

    best = (-1.0, None, None, None)  # (sil, model, k, labels)

    # -------- Stage 1: Coarse search --------
    coarse_scores = {}
    for k in coarse_grid:
        sil, model, labels = fit_and_score(k)
        if sil is None:
            continue
        coarse_scores[k] = sil
        if sil > best[0]:
            best = (sil, model, k, labels)

    if best[2] is None:
        raise ValueError("Coarse search found no valid k (check data or grid).")

    best_coarse_k = best[2]

    # -------- Stage 2: Fine search (± window) --------
    k_min = max(2, best_coarse_k - window)
    k_max = min(N - 1, best_coarse_k + window)
    fine_grid = list(range(k_min, k_max + 1))

    fine_scores = {}
    for k in fine_grid:
        sil, model, labels = fit_and_score(k)
        if sil is None:
            continue
        fine_scores[k] = sil
        if sil > best[0]:
            best = (sil, model, k, labels)
            
    best_score, best_model, best_k, best_labels = best

    return best_score, best_model, best_k, best_labels

def create_linkage(model, n_samples):

    counts = np.zeros(model.children_.shape[0])
    for i, merge in enumerate(model.children_):
        count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                count += 1  # leaf node
            else:
                count += counts[child_idx - n_samples]
        counts[i] = count

    linkage_matrix = np.column_stack([
        model.children_,      # which clusters were merged
        model.distances_,     # distance between merged clusters
        counts                # number of points in each new cluster
    ]).astype(float)

    return linkage_matrix




# Code for plotting tree structure

def plot_tree_interactive(root, leaf_nodes, figsize=(1200, 800), max_labels_display=10):
    """
    Plot the hierarchical tree structure with interactive hover for labels
    
    Parameters:
    - root: The root TreeNode of your hierarchical clustering
    - leaf_nodes: List of leaf TreeNodes
    - figsize: Figure size as tuple (width, height) in pixels
    - max_labels_display: Maximum number of labels to show on hover
    """
    leaf_set = set(leaf_nodes)
    
    # Calculate tree layout
    positions, levels = calculate_tree_positions(root, leaf_nodes)
    
    # Prepare data for plotting
    edge_trace = []
    node_traces = {'internal': [], 'leaf': []}
    
    # Collect edges and nodes
    def traverse_tree(node):
        node_id = id(node)
        x, y = positions[node_id]
        
        # Prepare hover text
        if node in leaf_set:
            labels = node.labels if node.labels else []
            hover_text = f"<b>Leaf Title: {node.summarry}</b><br><br>" if node.summarry else "<b>Leaf Node</b><br><br>"
            hover_text += f"Count: {node.count}<br>"
            hover_text += f"Distance: {node.distance:.3f}<br>"
            hover_text += f"Total labels: {len(labels)}<br>"
            hover_text += "<br><b>Sample Labels:</b><br>"
            
            for i, label in enumerate(labels[:max_labels_display], 1):
                # Wrap long labels
                label_str = str(label)
                if len(label_str) > 50:
                    label_str = label_str[:47] + "..."
                hover_text += f"{i}. {label_str}<br>"
            
            if len(labels) > max_labels_display:
                hover_text += f"<br><i>... and {len(labels) - max_labels_display} more</i>"
            
            node_type = 'leaf'
        else:
            hover_text = f"<b>Internal Node</b><br>"
            hover_text += f"Count: {node.count}<br>"
            hover_text += f"Distance: {node.distance:.3f}<br>"
            
            node_type = 'internal'
        
        node_traces[node_type].append({
            'x': x,
            'y': y,
            'text': hover_text,
            'customdata': node.count
        })
        
        # Draw edges to children
        if node not in leaf_set:
            if node.left:
                child_x, child_y = positions[id(node.left)]
                edge_trace.extend([
                    {'x': x, 'y': y},
                    {'x': child_x, 'y': child_y},
                    {'x': None, 'y': None}  # Break in line
                ])
                traverse_tree(node.left)
            
            if node.right:
                child_x, child_y = positions[id(node.right)]
                edge_trace.extend([
                    {'x': x, 'y': y},
                    {'x': child_x, 'y': child_y},
                    {'x': None, 'y': None}  # Break in line
                ])
                traverse_tree(node.right)
    
    traverse_tree(root)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add edges
    if edge_trace:
        edge_x = [point['x'] for point in edge_trace]
        edge_y = [point['y'] for point in edge_trace]
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none',
            showlegend=False,
            name='edges'
        ))
    
    # Add internal nodes
    if node_traces['internal']:
        internal_x = [n['x'] for n in node_traces['internal']]
        internal_y = [n['y'] for n in node_traces['internal']]
        internal_text = [n['text'] for n in node_traces['internal']]
        internal_labels = [str(n['customdata']) for n in node_traces['internal']]
        
        fig.add_trace(go.Scatter(
            x=internal_x,
            y=internal_y,
            mode='markers+text',
            marker=dict(
                size=15,
                color='lightblue',
                line=dict(color='darkblue', width=2)
            ),
            text=internal_labels,
            textposition='middle center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            hovertext=internal_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial'
            ),
            name='Internal Nodes'
        ))
    
    # Add leaf nodes
    if node_traces['leaf']:
        leaf_x = [n['x'] for n in node_traces['leaf']]
        leaf_y = [n['y'] for n in node_traces['leaf']]
        leaf_text = [n['text'] for n in node_traces['leaf']]
        leaf_labels = [str(n['customdata']) for n in node_traces['leaf']]
        
        fig.add_trace(go.Scatter(
            x=leaf_x,
            y=leaf_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='lightcoral',
                symbol='square',
                line=dict(color='darkred', width=2)
            ),
            text=leaf_labels,
            textposition='middle center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            hovertext=leaf_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial',
                align='left'
            ),
            name='Leaf Nodes'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Hierarchical Tree Structure',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis=dict(
            title='Node Position',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Tree Level',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            autorange='reversed'  # Root at top
        ),
        width=figsize[0],
        height=figsize[1],
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.show()
    return fig

def get_effective_leaves(root, min_distance_threshold=None):
    """
    Get all effective leaf nodes in the tree
    If min_distance_threshold is provided, nodes below this threshold are treated as leaves
    """
    leaves = []
    
    def traverse(node):
        if node.is_leaf():
            leaves.append(node)
        elif min_distance_threshold is not None and node.distance < min_distance_threshold:
            leaves.append(node)
        else:
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
    
    traverse(root)
    return leaves

def calculate_tree_positions(root, leaf_nodes):
    """Calculate x,y positions for each node in the tree"""
    positions = {}
    levels = {}
    leaf_set = set(leaf_nodes)
    
    # First pass: assign levels (depth from root)
    def assign_levels(node, level=0):
        levels[id(node)] = level
        if node not in leaf_set:
            if node.left:
                assign_levels(node.left, level + 1)
            if node.right:
                assign_levels(node.right, level + 1)
    
    assign_levels(root)
    
    # Second pass: assign horizontal positions
    leaf_counter = [0]  # Use list to make it mutable in nested function
    
    def assign_positions(node):
        if node in leaf_set:
            # This is an effective leaf
            positions[id(node)] = (leaf_counter[0], levels[id(node)])
            leaf_counter[0] += 1
            return leaf_counter[0] - 1
        
        left_pos = None
        right_pos = None
        
        if node.left:
            left_pos = assign_positions(node.left)
        if node.right:
            right_pos = assign_positions(node.right)
        
        # Position internal node at midpoint of children
        if left_pos is not None and right_pos is not None:
            x_pos = (left_pos + right_pos) / 2
        elif left_pos is not None:
            x_pos = left_pos
        elif right_pos is not None:
            x_pos = right_pos
        else:
            x_pos = leaf_counter[0]
            leaf_counter[0] += 1
        
        positions[id(node)] = (x_pos, levels[id(node)])
        return x_pos
    
    assign_positions(root)
    return positions, levels


def tree_to_linkage(
    root,
    sort_child_ids: bool = True
):

    # 1) Count leaves first to size Z.
    def count_leaves(node) -> int:
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    n_leaves = count_leaves(root)
    if n_leaves <= 1:
        # No merges → empty linkage
        return np.zeros((0, 4), dtype=float), {root: 0}, {}

    Z = np.zeros((n_leaves - 1, 4), dtype=float)

    # 2) Assign IDs via post-order: leaves first (0..n-1), internals next (n..2n-2).
    node_id_to_cluster_id: Dict[int, int] = {}
    node_id_to_row: Dict[int, int] = {}

    leaf_next = 0
    internal_next = n_leaves
    row_next = 0

    # Left-to-right DFS to assign leaf IDs deterministically
    def assign_leaf_ids(node):
        nonlocal leaf_next
        if node.is_leaf():
            node_id_to_cluster_id[node.idx] = leaf_next
            leaf_next += 1
        else:
            assign_leaf_ids(node.left)
            assign_leaf_ids(node.right)

    assign_leaf_ids(root)

    # Post-order traversal builds Z rows and assigns internal cluster IDs
    def build_rows(node) -> int:
        nonlocal internal_next, row_next

        if node.is_leaf():
            return node_id_to_cluster_id[node.idx]

        # Process children first
        left_id = build_rows(node.left)
        right_id = build_rows(node.right)

        # Optional: enforce left_id < right_id for readability
        if sort_child_ids and left_id > right_id:
            left_id, right_id = right_id, left_id

        # Decide cluster size (4th column)

        # Record this merge into Z at current row
        Z[row_next, 0] = float(left_id)
        Z[row_next, 1] = float(right_id)
        Z[row_next, 2] = float(node.distance)   # merge height
        Z[row_next, 3] = float(node.count)

        # Assign this node's cluster ID = n_leaves + row_next
        node_id_to_cluster_id[node.idx] = internal_next
        node_id_to_row[node.idx] = row_next

        internal_next += 1
        row_next += 1

        return node_id_to_cluster_id[node.idx]

    build_rows(root)

    return Z, node_id_to_cluster_id, node_id_to_row


privacy_taxonomy = {

    "Data Collection": [
        "Collecting unnecessary personal data",
        "Purpose of data collection",
        "Data Aggregation",
        "Data Minimization",
    ],

    "Data Sharing": [
        "Accidental sharing",
        "Forced sharing",
        "Unintended sharing",
        "Cross-app sharing",
        "Secondary use",
        "Calendar sharing",
        "Password protected sharing"
    ],

    "Data Deletion": [
        "Browsing history removal",
        "Delete Video history",
        "Delete search history",
        "Cookie removal",
        "Photo deletion",
        "Video deletion",
        "File deletion",
        "Profile deletion"
    ],

    "Remove Personally Identifiable Information": [
        "Remove name",
        "Remove address",
        "Remove phone number",
        "Remove Personal information",
        "Remove photo"
    ],

    "Data Exposure": [
        "To advertisers",
        "To app developers",
        "Public accessibility",
        "Data disclosure",
        "Safety",
        "Use Limitation"
    ],

    "Data Hiding": [
        "Name",
        "Video",
        "Events",
        "Albums",
        "Photos",
        "Contacts",
        "Display name",
        "Email",
        "Messages",
        "Notes",
        "Playlist",
        "Folders",
        "Meetings"
    ],

    "Location and Tracking": [
        "Location data",
        "Tracking"
    ],

    "Consent": [
        "Giving consent",
        "Agreement",
        "Authorize",
        "Consent process",
        "Opt-out (notice awareness)",
        "Without consent",
        "Mis-activation",
        "Forced consent"
    ],

    "Privacy Controls": [
        "Privacy settings",
        "Changing personal information",
        "Location sharing",
        "Download my data",
        "Parental controls",
        "Privacy defaults",
        "Password protected controls"
    ],

    "Anonymity / Identification": [
        "User wants to be anonymous",
        "Incognito",
        "Hide group participants",
        "Fear of identification",
        "Misattribution"
    ],

    "Advertising": [
        "Personalized advertising",
        "Paid services"
    ],

    "Data Security": [
        "Safety",
        "Protection",
        "Breach of confidentiality",
        "Data breach",
        "Account hacking",
        "Fake profiles"
    ],

    "Password Issues": [],

    "Data Accuracy": [
        "Inaccurate",
        "Obsolete",
        "Wrong association"
    ],

    "Safety": [
        "Blackmail",
        "Appropriation",
        "Intrusion",
        "Decisional interference"
    ],

    "Selling data": [
        "To 3rd parties",
        "For ads purposes"
    ],

    "Surveillance": [
        "Spying",
        "Stalkerware",
        "Satellite"
    ],

    "Privacy Invasion": [],

    "Privacy policies and laws": [],

    "Positive Privacy": []
}
