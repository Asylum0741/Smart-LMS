from graphviz import Digraph

def linear_arrow_to_tree(text):
    nodes = [n.strip() for n in text.split("→")]
    tree = {
        "root": nodes[0],
        "children": []
    }

    current = tree
    for n in nodes[1:]:
        child = {
            "node": n,
            "children": []
        }
        current["children"].append(child)
        current = child

    return tree

def render_tree(tree):
    dot = Digraph()
    root = tree['root']
    dot.node("root", root)

    def add_children(parent_id, children):
        for idx, child in enumerate(children):
            child_id = f"{parent_id}_{idx}"
            dot.node(child_id, child["node"])
            dot.edge(parent_id, child_id)
            add_children(child_id, child["children"])

    add_children("root", tree['children'])
    return dot
