import json
import numpy as np
from anytree import AnyNode, NodeMixin, findall_by_attr


def add_node_to_union_class(union_root, node_to_add, parent_name):
    # Find the parent in the union tree
    parent = None
    if parent_name:
        parent = findall_by_attr(union_root, parent_name, name="name")[0]
    else:  # If no parent name, it means this is the root
        parent = union_root
    
    # Add the node if it doesn't already exist
    if not findall_by_attr(union_root, node_to_add.name, name="name"):
        AnyNode(name=node_to_add.name, parent=parent)

def traverse_and_add_class(union_root, current_node, parent_name=None):
    add_node_to_union_class(union_root, current_node, parent_name)
    for child in current_node.children:
        traverse_and_add_class(union_root, child, current_node.name)


def add_node_to_union_instance(union_root, node_to_add, count, parent_name):
    # Find the parent in the union tree
    parent = None
    if parent_name:
        parent = findall_by_attr(union_root, parent_name, name="name")[0]
    else:   # If no parent name, it means this is the root
        parent = union_root

    # Add the node if it doesn't already exist
    if not findall_by_attr(union_root, f'{node_to_add.name}_{count}', name='name'):
        AnyNode(name=f'{node_to_add.name}_{count}', parent=parent)


def traverse_and_add_instance(union_root, current_node, node_occur, parent_name=None):
    node_occur[current_node.name] += 1
    add_node_to_union_instance(union_root, current_node,
                               node_occur[current_node.name],
                               parent_name)
    for child in current_node.children:
        traverse_and_add_instance(union_root, child, node_occur,
                                  f'{current_node.name}_{node_occur[current_node.name]}')


def recon_tree(adj, node_names):
    # reconstruct a tree that is a subset of the union tree
    recon_root = AnyNode(name=node_names[0])
    def build_tree(parent, parent_idx, adj):
        child_indices = np.argwhere(adj[parent_idx] == 1).flatten().tolist()
        if len(child_indices) == 0:
            return
        else:
            for idx in child_indices:
                child = AnyNode(name=node_names[idx], parent=parent)
                build_tree(child, idx, adj)
    build_tree(recon_root, 0, adj)
    return recon_root


class OriNode(NodeMixin):
    """Original node (before merging)
    """
    def __init__(self, ori_id, name, parent=None, children=None):
        super(OriNode, self).__init__()
        self.ori_id = ori_id    # id before merging
        self.name = name        # name before merging
        self.parent = parent
        if children:
            self.children = children

    def __repr__(self) -> str:
        return f"OriNode(ori_id={self.ori_id}, name={self.name})"
    
    def __str__(self) -> str:
        return f"OriNode(ori_id={self.ori_id}, name={self.name})"

    def is_leaf_node(self):
        return len(self.children) == 0

    def get_ids_of_all_children(self):
        assert not self.is_leaf_node()
        ori_ids = []

        def traverse(node: OriNode):
            if not node.is_leaf_node():
                for child in node.children:
                    ori_ids.append(child.ori_id)
                    traverse(child)
        
        traverse(self)
        return ori_ids



class AMNode(NodeMixin):
    """After merging node
    """
    def __init__(self, ori_id, id, objs, name, parent=None, children=None):
        super(AMNode, self).__init__()
        self.ori_id = ori_id    # id before merging
        self.id = id            # id after merging
        self.objs = objs
        self.name = name
        self.parent = parent
        if self.children:
            self.children = children
    
    def __repr__(self) -> str:
        return f"AMNode(ori_id={self.ori_id}, id={self.id}, objs={len(self.objs)},"\
            + f" name={self.name})"
    
    def __str__(self) -> str:
        return f"AMNode(ori_id={self.ori_id}, id={self.id}, objs={len(self.objs)},"\
            + f" name={self.name})"

    def is_leaf_node(self):
        return len(self.children) == 0
    

class NewNode(NodeMixin):   
    """New node
    """
    def __init__(self, ori_id, id, new_id, objs, name,
                 parent=None, children=None):
        super(NewNode, self).__init__()
        self.ori_id = ori_id    # id before merging
        self.id = id            # id after merging
        self.new_id = new_id    
        self.objs = objs
        self.name = name
        self.parent = parent
        if self.children:
            self.children = children
    
    def __repr__(self) -> str:
        return f"NewNode(ori_id={self.ori_id}, id={self.id}, new_id={self.new_id}"\
            + f" objs={self.objs}, name={self.name})"
    
    def __str__(self) -> str:
        return f"NewNode(ori_id={self.ori_id}, id={self.id}, new_id={self.new_id}"\
            + f" objs={self.objs}, name={self.name})"

    def is_leaf_node(self):
        return len(self.children) == 0


def build_tree_from_json(json_path):
    """This only works with result.json (before merging)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)[0]

    all_anynodes = []
    ori_id_to_list_idx = {}
    
    def traverse(node, parent):
        ori_id_to_list_idx[node['id']] = len(all_anynodes)
        new = OriNode(ori_id=node['id'], name=node['name'])
        new.parent = parent
        all_anynodes.append(new)

        if 'children' in node and node['children']:
            for child in node['children']:
                traverse(child, new)
        return

    # ori_ids.append(data['id'])
    ori_id_to_list_idx[data['id']] = len(all_anynodes)
    all_anynodes.append(OriNode(ori_id=data['id'], name=data['name']))

    assert data['children']
    for node in data['children']:
        traverse(node, all_anynodes[0])

    return all_anynodes, ori_id_to_list_idx


def build_tree_from_json_after_merge(json_path):
    """This only works with result_after_merging.json (after merging)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)[0]

    # ori_ids = []
    all_anynodes = []
    ori_id_to_list_idx = {}
    
    def traverse(node, parent):
        ori_id_to_list_idx[node['id']] = len(all_anynodes)
        new = AMNode(ori_id=node['ori_id'],
                        id=node['id'],
                        objs=node['objs'],
                        name=node['name'])
        new.parent = parent
        all_anynodes.append(new)
        # ori_ids.append(node['id'])

        if 'children' in node and node['children']:
            for child in node['children']:
                traverse(child, new)
        return

    # ori_ids.append(data['id'])
    ori_id_to_list_idx[data['id']] = len(all_anynodes)
    all_anynodes.append(AMNode(ori_id=data['id'],
                               id=data['id'],
                               objs=data['objs'],
                               name=data['name']))

    assert data['children']
    for node in data['children']:
        traverse(node, all_anynodes[0])

    return all_anynodes, ori_id_to_list_idx


def find_all_children(node: AnyNode):
    children = []
    
    def traverse(node, children):
        if node.children != None:
            children += list(node.children)
            for child in node.children:
                traverse(child, children)
    
    traverse(node, children)

    return children


def is_leaf_node(node: AnyNode):
    return node.children == None
