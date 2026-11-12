from dataclasses import dataclass, field
from typing import List
import plotly.graph_objects as go

def get_effective_leaves_tax(root, min_distance_threshold=None):
    """
    Get all effective leaf nodes in the n-ary tree
    If min_distance_threshold is provided, nodes below this threshold are treated as leaves
    """
    leaves = []
    
    def traverse(node):
        if node.is_leaf():
            leaves.append(node)
        elif min_distance_threshold is not None and node.distance < min_distance_threshold:
            leaves.append(node)
        else:
            for child in node.children:
                traverse(child)
    
    traverse(root)
    return leaves

def calculate_tree_positions(root, leaf_nodes):
    """Calculate x,y positions for each node in the n-ary tree"""
    positions = {}
    levels = {}
    leaf_set = set(id(node) for node in leaf_nodes)
    
    # First pass: assign levels (depth from root)
    def assign_levels(node, level=0):
        levels[id(node)] = level
        if id(node) not in leaf_set:
            for child in node.children:
                assign_levels(child, level + 1)
    
    assign_levels(root)
    
    # Second pass: assign horizontal positions
    leaf_counter = [0]  # Use list to make it mutable in nested function
    
    def assign_positions(node):
        if id(node) in leaf_set:
            # This is an effective leaf
            positions[id(node)] = (leaf_counter[0], levels[id(node)])
            leaf_counter[0] += 1
            return leaf_counter[0] - 1
        
        child_positions = []
        
        # Get positions of all children
        for child in node.children:
            child_pos = assign_positions(child)
            child_positions.append(child_pos)
        
        # Position internal node at center of children
        if child_positions:
            x_pos = sum(child_positions) / len(child_positions)
        else:
            # Node with no children (shouldn't happen if leaf_set is correct)
            x_pos = leaf_counter[0]
            leaf_counter[0] += 1
        
        positions[id(node)] = (x_pos, levels[id(node)])
        return x_pos
    
    assign_positions(root)
    return positions, levels

def plot_tree_interactive_tax(root, leaf_nodes, figsize=(1200, 800), max_labels_display=10):
    """
    Plot the hierarchical n-ary tree structure with interactive hover for labels
    
    Parameters:
    - root: The root TaxonNode of your hierarchical clustering
    - leaf_nodes: List of leaf TaxonNodes
    - figsize: Figure size as tuple (width, height) in pixels
    - max_labels_display: Maximum number of labels to show on hover
    """
    leaf_set = set(id(node) for node in leaf_nodes)
    
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
        if node_id in leaf_set:
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
            hover_text = f"<b>Node Title: {node.summarry}</b><br><br>" if node.summarry else "<b>Internal Node</b><br><br>"
            hover_text += f"Count: {node.count}<br>"
            hover_text += f"Distance: {node.distance:.3f}<br>"
            hover_text += f"Children: {len(node.children)}<br>"
            
            node_type = 'internal'
        
        node_traces[node_type].append({
            'x': x,
            'y': y,
            'text': hover_text,
            'customdata': node.count
        })
        
        # Draw edges to all children
        if node_id not in leaf_set:
            for child in node.children:
                child_x, child_y = positions[id(child)]
                edge_trace.extend([
                    {'x': x, 'y': y},
                    {'x': child_x, 'y': child_y},
                    {'x': None, 'y': None}  # Break in line
                ])
                traverse_tree(child)
    
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
            'text': 'Interactive Hierarchical N-ary Tree Structure',
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
    
    return fig