from promptLLM import open_route_ai_models, openai_chatgpt_models
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def get_coverage(label_embeddings, leaf_embeddings, threshold: float = 0.6) -> dict:
    """
    Fraction of pseudo-labels whose best cosine similarity to any taxonomy leaf
    is >= threshold.

    Both embedding matrices must be L2-normalized (shape: n x d).
    """
    S = label_embeddings @ leaf_embeddings.T   # (n_labels, n_leaves)
    best_scores = S.max(axis=1)                # (n_labels,)
    covered = int((best_scores >= threshold).sum())
    coverage = covered / len(label_embeddings)
    return {
        "coverage": coverage,
        "covered": covered,
        "total_labels": len(label_embeddings),
        "threshold": threshold,
    }

def process_single(input_str):
    return openai_chatgpt_models(
        prompt=input_str,
        model_name="openai/gpt-4o"
    )

def get_path_granularity(paths: list) -> list:
    def get_prompt(path):
        return (
            "Concepts are naturally organized in multi-dimensional taxonomic structures, with more specific concepts being the children of a broader topic.\n\n"
            f"Given the root topic: 'Privacy/Securrity', decide whether this path from the taxonomy has good granularity: '{path}' Check whether the child node is a more specific subaspect of the parent node. \n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )
    input_strs = [get_prompt(path) for path in paths]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)
        
        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()
    
    results = []
    for path, output in zip(paths, outputs):
        result = {"path": path}
        if "<good granularity>" in output.lower():
            result['score'] = 1
        elif "<bad granularity>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results


def get_parent_child_granularity(pairs: list) -> list:
    """Score granularity for each parent-child pair individually."""
    def get_prompt(pair):
        parent = pair['parent']
        child = pair['child']
        return (
            "Concepts are naturally organized in multi-dimensional taxonomic structures, with more specific concepts being the children of a broader topic.\n\n"
            f"Given the root topic: 'Privacy/Security', decide whether this parent-child relationship has good granularity:\n"
            f"  Parent: '{parent}'\n"
            f"  Child: '{child}'\n\n"
            f"Check whether '{child}' is a more specific subaspect of '{parent}'.\n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )

    input_strs = [get_prompt(pair) for pair in pairs]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)

        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()

    results = []
    for pair, output in zip(pairs, outputs):
        result = {"parent": pair['parent'], "child": pair['child']}
        if "<good granularity>" in output.lower():
            result['score'] = 1
        elif "<bad granularity>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)

    return results


def get_level_granularity(levels: list) -> list:
    def get_prompt(level_instance):
        parent = level_instance['parent']
        siblings = level_instance['siblings']
        return (
            f"You are determining the coherence of a set of 'Privacy/Securrity' subtopics of the parent topic {parent}.\n\nThe parent topic is: {parent}.\n\nThe set of siblings, which are child subtopics of the parent, are: '{', '.join(siblings)}'\n\nEvaluate the overall coherence of the sibling set based on their collective specificity and granularity relative to {parent}. Use the following scoring criteria:\n\n"
            f"Score=<no_sibling_coherence>: The set is highly inconsistent or incoherent (only one subtopic), with most topics significantly misaligned in specificity relative to the parent.\n"
            f"Score=<weak_sibling_coherence>: The set shows considerable inconsistency, with several topics deviating noticeably from the expected level of specificity.\n"
            f"Score=<reasonable_sibling_coherence>: The set is generally coherent, with only minor inconsistencies in specificity among the topics.\n"
            f"Score=<strong_sibling_coherence>: The set is fully coherent, with all topics properly matching the expected level of specificity and granularity for the parent.\n"
            "Output options: '<no_sibling_coherence>', '<weak_sibling_coherence>', '<reasonable_sibling_coherence>', or '<strong_sibling_coherence>'. Do some simple rationalization before giving the output if possible."
        )

    input_strs = [get_prompt(level) for level in levels]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)
        
        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"path": level_instance}
        if "<no_sibling_coherence>" in output.lower():
            result['score'] = 0
        elif "<weak_sibling_coherence>" in output.lower():
            result['score'] = 1/3
        elif "<reasonable_sibling_coherence>" in output.lower():
            result['score'] = 2/3
        elif "<strong_sibling_coherence>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results


# I need some sort of metrics that tell me compared to the original taxonomy what novel and new concepts were discovered in the new taxonomy.

def get_leaf_novelty(leaves: list) -> list:
    """
    Assess novelty for each leaf node independently (no parent context).
    `leaves` is a list of leaf name strings.
    """
    def get_prompt(leaf):
        return (
            f"You are evaluating a Privacy/Security taxonomy. Assess whether the leaf topic '{leaf}' "
            f"is a novel and meaningful concept in the Privacy/Security domain.\n\n"
            f"Context: Common, expected leaf topics in a standard Privacy/Security taxonomy are broad, "
            f"obvious categories that any standard taxonomy would include. Novel leaf topics go beyond "
            f"these; they surface specific, underrepresented, or emerging concerns.\n\n"
            f"Scoring criteria:\n"
            f"  <not_novel>: Redundant or obvious — largely covered by standard Privacy/Security taxonomy "
            f"leaves, or too vague to add taxonomic value.\n"
            f"  <highly_novel>: Surfaces a specific, underrepresented, or emerging concern (e.g., a particular "
            f"data type, demographic vulnerability, or institutional practice) that meaningfully enriches the "
            f"taxonomy in ways a standard categorization would miss.\n\n"
            f"Instructions:\n"
            f"1. Briefly state what common leaf topics in Privacy/Security taxonomies typically look like.\n"
            f"2. Explain what '{leaf}' adds (or doesn't add) beyond those.\n"
            f"3. Output exactly one of: <not_novel>, or <highly_novel>."
        )

    input_strs = [get_prompt(leaf) for leaf in leaves]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)

        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()

    results = []
    for leaf, output in zip(leaves, outputs):
        result = {"leaf": leaf}
        if "<not_novel>" in output.lower():
            result['score'] = 0
        elif "<highly_novel>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)

    return results


def get_node_novelty(nodes: list) -> list:
    """
    Assess novelty for each taxonomy node (leaf or internal) independently.
    `nodes` is a list of node name strings.
    """
    def get_prompt(node):
        return (
            f"You are evaluating a Privacy/Security taxonomy. Assess whether the topic '{node}' "
            f"is a novel and meaningful concept in the Privacy/Security domain.\n\n"
            f"Context: Common, expected topics in a standard Privacy/Security taxonomy are broad, "
            f"obvious categories that any standard taxonomy would include. Novel topics go beyond "
            f"these; they surface specific, underrepresented, or emerging concerns.\n\n"
            f"Scoring criteria:\n"
            f"  <not_novel>: Redundant or obvious — largely covered by standard Privacy/Security taxonomy "
            f"topics, or too vague to add taxonomic value.\n"
            f"  <highly_novel>: Surfaces a specific, underrepresented, or emerging concern (e.g., a particular "
            f"data type, demographic vulnerability, or institutional practice) that meaningfully enriches the "
            f"taxonomy in ways a standard categorization would miss.\n\n"
            f"Instructions:\n"
            f"1. Briefly state what common topics in Privacy/Security taxonomies typically look like.\n"
            f"2. Explain what '{node}' adds (or doesn't add) beyond those.\n"
            f"3. Output exactly one of: <not_novel>, or <highly_novel>."
        )

    input_strs = [get_prompt(node) for node in nodes]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)

        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()

    results = []
    for node, output in zip(nodes, outputs):
        result = {"node": node}
        if "<not_novel>" in output.lower():
            result['score'] = 0
        elif "<highly_novel>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)

    return results



def get_novelty(levels: list) -> list:
    def get_prompt(level_instance):
        parent = level_instance['parent']
        child = level_instance['child']
        return (
            f"You are evaluating a Privacy/Security taxonomy. Assess whether the child topic '{child}' "
            f"is a novel and meaningful subtopic of '{parent}'.\n\n"
            f"Context: Common, expected subtopics under '{parent}' are things like broad, obvious categories "
            f"that any standard taxonomy would include. Novel subtopics go beyond these; they surface "
            f"specific, underrepresented, or emerging concerns within the Privacy/Security domain.\n\n"
            f"Scoring criteria:\n"
            f"  <not_novel>: Redundant or obvious — largely covered by other typical subtopics under '{parent}', "
            f"or too vague to add taxonomic value.\n"
            f"  <highly_novel>: Surfaces a specific, underrepresented, or emerging concern (e.g., a particular "
            f"data type, demographic vulnerability, or institutional practice) that meaningfully enriches the "
            f"taxonomy in ways a standard categorization would miss.\n\n"
            f"Instructions:\n"
            f"1. Briefly state what common subtopics under '{parent}' typically look like.\n"
            f"2. Explain what '{child}' adds (or doesn't add) beyond those.\n"
            f"3. Output exactly one of: <not_novel>, or <highly_novel>."
        )
        
    input_strs = [get_prompt(level) for level in levels]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)
        
        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"level": level_instance}
        if "<not_novel>" in output.lower():
            result['score'] = 0
        elif "<highly_novel>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results



"""
def get_novelty(levels: list) -> list: 
    def get_prompt(level_instance):
        parent = level_instance['parent']
        child = level_instance['child']
        return (
            f"You are evaluating a Privacy/Security taxonomy. Assess whether the child topic '{child}' "
            f"is a novel and meaningful subtopic of '{parent}'.\n\n"
            f"Context: Common, expected subtopics under '{parent}' are things like broad, obvious categories "
            f"that any standard taxonomy would include. Novel subtopics go beyond these; they surface "
            f"specific, underrepresented, or emerging concerns within the Privacy/Security domain.\n\n"
            f"Scoring criteria:\n"
            f"  <not_novel>: Redundant or obvious — largely covered by other typical subtopics under '{parent}', "
            f"or too vague to add taxonomic value.\n"
            f"  <somewhat_novel>: Adds a distinct angle or specificity not commonly found in standard taxonomies, "
            f"but the insight is incremental rather than significant.\n"
            f"  <highly_novel>: Surfaces a specific, underrepresented, or emerging concern (e.g., a particular "
            f"data type, demographic vulnerability, or institutional practice) that meaningfully enriches the "
            f"taxonomy in ways a standard categorization would miss.\n\n"
            f"Instructions:\n"
            f"1. Briefly state what common subtopics under '{parent}' typically look like.\n"
            f"2. Explain what '{child}' adds (or doesn't add) beyond those.\n"
            f"3. Output exactly one of: <not_novel>, <somewhat_novel>, or <highly_novel>."
        )
        
    input_strs = [get_prompt(level) for level in levels]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single, s): i for i, s in enumerate(input_strs)}
        outputs = [None] * len(input_strs)
        
        for future in tqdm(as_completed(futures), total=len(input_strs)):
            idx = futures[future]
            outputs[idx] = future.result()
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"level": level_instance}
        if "<not_novel>" in output.lower():
            result['score'] = 0
        elif "<somewhat_novel>" in output.lower():
            result['score'] = 1/2
        elif "<highly_novel>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results
"""