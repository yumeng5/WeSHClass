

class ClassNode(object):
    def __init__(self, name, parent, label=None):
        self.name = name
        self.parent = parent
        self.label = label
        self.children = []
        self.keywords = []
        self.expanded = []
        self.doc_idx = []
        self.model = None
        self.embedding = None
        self.sup_idx = []

    def add_child(self, node):
        self.children.append(node)

    def add_keywords(self, keywords):
        self.keywords += keywords

    def find_descendants(self):
        if self.children == []:
            return []
        else:
            descendants = self.children
            for child in self.children:
                descendants += child.find_descendants()
            return descendants

    def find_leaves(self):
        leaves = []
        if self.children == []:
            leaves += [self]
        else:
            for child in self.children:
                leaves += child.find_leaves()
        return leaves

    def find_ancestors(self):
        if self.label == -1 or self.parent.label == -1 : # self or parent is ROOT
            return []
        return [self.parent] + self.parent.find_ancestors()

    def get_full_label(self):
        full_label = [self.label]
        ancestors = self.find_ancestors()
        for ancestor in ancestors:
            full_label.append(ancestor.label)
        return full_label

    def get_size(self):
        sz = 1
        for child in self.children:
            sz += child.get_size()
        return sz

    def get_height(self):
        if self.children == []:
            return 0
        else:
            heights = [child.get_height() for child in self.children]
            return max(heights) + 1

    def find(self, name):
        if type(name) == str:
            if name == self.name:
                return self
        elif type(name) == int:
            if name == self.label:
                return self
        if self.children == []:
            return None
        for child in self.children:
            if child.find(name):
                return child.find(name)
        return None
    
    def find_add_child(self, name, node):
        target = self.find(name)
        assert target
        target.add_child(node)

    def find_add_keywords(self, name, keywords):
        target = self.find(name)
        assert target, f'Class {name} not found!'
        target.add_keywords(keywords)

    def aggregate_keywords(self):
        if self.children == []:
            assert self.keywords
        else:
            if self.keywords == []:
                for child in self.children:
                    self.add_keywords(child.aggregate_keywords())
        return self.keywords

    def name2label(self, name):
        target = self.find(name)
        assert target
        return target.get_full_label()

    def find_at_level(self, level):
        targets = []
        if level == 0:
            targets.append(self)
        else:
            for child in self.children:
                targets += child.find_at_level(level-1)
        return targets

    def siblings_at_level(self, level):
        siblings_map = {}
        parent_nodes = self.find_at_level(level)
        offset = 0
        for node in parent_nodes:
            num_children = len(node.children)
            siblings = range(offset, offset+num_children)
            for i in range(offset, offset+num_children):
                siblings_map[i] = siblings
            offset += num_children
        return siblings_map

    def visualize_tree(self):
        print_string = self.name + ' (' + str(self.label) + ') ' + '\t'
        print_string += ','.join(child.name for child in self.children) + '\n'
        for child in self.children:
            print_string += child.visualize_tree()
        return print_string

    def visualize_node(self):
        print_string = self.name + ' (' + str(self.label) + ') ' + '\n'
        if self.parent:
            print_string += "Parent: " + self.parent.name + '\n'
        else:
            print_string += "Parent: None \n"
        if self.children:
            print_string += "Children: " + ','.join(child.name for child in self.children) + '\n'
        else:
            print_string += "Children: None \n"
        if self.keywords:
            print_string += "Keywords: " + ','.join(keyword for keyword in self.keywords) + '\n'
        else:
            print_string += "Keywords: None \n"
        print_string += '\n'
        return print_string

    def visualize_nodes(self):
        print_string = self.visualize_node()
        for child in self.children:
            print_string += child.visualize_nodes()
        return print_string
    