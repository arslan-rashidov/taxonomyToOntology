import numpy as np
from nltk.corpus import wordnet as wn

from fastText.python.fasttext_module import fasttext
from fastText.python.fasttext_module.fasttext.util import util

from anytree import Node

from anytree.exporter import DotExporter

"""
Initial settings setup
"""
util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

start = Node("ASOS_Shopping")

male = Node("male", parent=start)
polo = Node("polo", parent=male)
polo_color1 = Node("green", parent=polo)
polo_color2 = Node("blue", parent=polo)
polo_size1 = Node("large", parent=polo)
hoodie = Node("hoodie", parent=male)
hoodie_color1 = Node("green", parent=hoodie)
hoodie_color2 = Node("black", parent=hoodie)
hoodie_size1 = Node("small", parent=hoodie)
hoodie_size2 = Node("large", parent=hoodie)
shorts = Node("shorts", parent=male)
shorts_color1 = Node("blue", parent=shorts)
shorts_size1 = Node("medium", parent=shorts)
shorts_size2 = Node("large", parent=shorts)

female = Node("female", parent=start)
dress = Node("dress", parent=female)
dress_color1 = Node("white", parent=dress)
dress_color2 = Node("blue", parent=dress)
dress_size1 = Node("large", parent=dress)
dress_size2 = Node("medium", parent=dress)
skirt = Node("skirt", parent=female)
skirt_color1 = Node("brown", parent=skirt)
skirt_color2 = Node("red", parent=skirt)
skirt_size1 = Node("small", parent=skirt)
blouse = Node("blouse", parent=female)
blouse_color1 = Node("white", parent=blouse)
blouse_size1 = Node("small", parent=blouse)
blouse_size2 = Node("medium", parent=blouse)

# DotExporter(start).to_picture("udo.png")

words_list = []
for synset in wn.all_synsets():
    word = str(synset.lemmas()[0].name())
    words_list.append(word)

words_vectors = {w: ft.get_sentence_vector(w) for w in words_list}


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    (https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    multi = norm_a * norm_b
    result = dot_product / multi
    if str(result) == "nan":
        return float(0)
    return float(dot_product / multi)


def compare_word(w, words_vectors):
    """
    Compares new word with those in the words vectors dictionary
    """
    vec = ft.get_sentence_vector(w)
    return {w1: cos_sim(vec, vec1) for w1, vec1 in words_vectors.items()}


def compare_2words(w1, w2):
    """
    Compares 2 words
    """
    vec1 = ft.get_sentence_vector(w1)
    vec2 = ft.get_sentence_vector(w2)

    return cos_sim(vec1, vec2)


def distance2vote(d, a=3, b=5):
    """
    Returns the weight of the vote based on the distance.
    """
    sim = np.maximum(0, 1 - d ** 2 / 2)
    return np.exp(-d ** a) * sim ** b


def get_groups_in_children(children_groups_merge):
    """
    Get groups based on similarities between words.
    """
    if len(children_groups_merge) == 1:
        return {0: [children_groups_merge[0]]}

    children_copy = children_groups_merge.copy()
    groups = {}
    id = 0

    while len(children_copy) != 0:
        node1 = children_groups_merge[0]
        groups[id] = [node1]
        if len(children_copy) == 1:
            break
        for i in range(1, len(children_groups_merge)):
            node2 = children_groups_merge[i]

            similarity_between_nodes = compare_2words(node1.name, node2.name)
            if similarity_between_nodes > 0.27:
                groups[id].append(node2)
                children_copy.remove(node2)
        children_copy.remove(node1)
        children_groups_merge = children_copy.copy()

        id += 1
    return groups


def get_children_for_group(group):
    """
    Get all children of every word in a group.
    """
    nodes = []
    for node in group:
        for children in node.children:
            nodes.append(children)
    return nodes


def get_group_hyperonym(group):
    """
    Get best word describing a group(hyperonym).
    """
    all_votes = []
    for node in group:
        votes = {}

        similarity = compare_word(node.name, words_vectors)
        similarity = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1])}

        for lemma_name in list(similarity)[-100:]:
            distance = similarity[lemma_name]
            for hyper in wn.synsets(lemma_name)[0].hypernyms():
                for lemma in hyper.lemmas():
                    if lemma.name() not in votes.keys():
                        votes[lemma.name()] = 0
                    votes[lemma.name()] += distance2vote(distance)
        all_votes.append({k: v for k, v in sorted(votes.items(), key=lambda item: item[1])})

    frequent_votes = {}

    for vote in all_votes:
        vote_count = len(vote.keys())
        for i in range(vote_count - 1, vote_count - 16, -1):
            lemma_name = list(vote.keys())[i]
            votes_received = vote[lemma_name]
            if lemma_name not in list(frequent_votes.keys()):
                frequent_votes[lemma_name] = 0
            frequent_votes[lemma_name] += votes_received
    frequent_votes = {k: v for k, v in sorted(frequent_votes.items(), key=lambda item: item[1])}

    def decide_hyperonym(frequent_votes, group):
        print("Группа состоит из следующих слов: ")
        for node in group:
            print(node.name, end=" ")
        print("\nКак бы вы ее назвали?")
        for h_i in range(0, 8):
            hyperonym = list(frequent_votes.keys())[len(list(frequent_votes.keys())) - 1 - h_i]
            score = frequent_votes[hyperonym]
            print(f"{h_i}. {hyperonym} - {score}")
        return list(frequent_votes.keys())[len(list(frequent_votes.keys())) - 1 - int(input("Выберите цифру: "))]

    hyperonym = decide_hyperonym(frequent_votes, group)

    return hyperonym


def transform_taxonomy_to_ontology(start_node):
    """
    Transfrom a taxonomy(by its head node) to a ontology.
    """
    start = Node(str(start_node.name))
    tree_is_transformed = False
    parent_node = start
    children_groups_merge = list(start_node.children)
    grandsons_nodes = get_children_for_group(children_groups_merge)
    while tree_is_transformed is False:
        groups = get_groups_in_children(children_groups_merge)

        new_nodes = []
        new_nodes_names = []

        for group in list(groups.values()):
            group_hyperonym = get_group_hyperonym(group)
            if group_hyperonym not in new_nodes_names:
                new_node = Node(group_hyperonym, parent=parent_node)
                new_nodes.append(new_node)
                new_nodes_names.append(group_hyperonym)
        if len(new_nodes) == 1:
            parent_node = new_nodes[0]
        children_groups_merge = grandsons_nodes
        grandsons_nodes = get_children_for_group(children_groups_merge)

        if len(children_groups_merge) == 0:
            tree_is_transformed = True

    return start


def main():
    """
    main() method.
    """
    ontology_start_node = transform_taxonomy_to_ontology(start_node=start)

    DotExporter(ontology_start_node).to_picture("finish.png")


if __name__ == '__main__':
    main()
