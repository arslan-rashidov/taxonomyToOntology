import numpy as np
from nltk.corpus import wordnet as wn

# download English pretrained model
from fastText.python.fasttext_module import fasttext
from fastText.python.fasttext_module.fasttext.util import util

from anytree import Node, RenderTree

from anytree.exporter import DotExporter
# graphviz needs to be installed for the next line!

start = Node("ASOS_Shopping")

male = Node("Man", parent=start)
polo = Node("Polo", parent=male)
polo_color1 = Node("green", parent=polo)
polo_color2 = Node("blue", parent=polo)
polo_size1 = Node("large", parent=polo)
hoodie = Node("Hoodie", parent=male)
hoodie_color1 = Node("green", parent=hoodie)
hoodie_color2 = Node("black", parent=hoodie)
hoodie_size1 = Node("small", parent=hoodie)
hoodie_size2 = Node("large", parent=hoodie)
shorts = Node("Shorts", parent=male)
shorts_color1 = Node("blue", parent=shorts)
shorts_size1 = Node("medium", parent=shorts)
shorts_size2 = Node("large", parent=shorts)


female = Node("Woman", parent=start)
dress = Node("Dress", parent=female)
dress_color1 = Node("white", parent=dress)
dress_color2 = Node("blue", parent=dress)
dress_size1 = Node("large", parent=dress)
dress_size2 = Node("medium", parent=dress)
skirt = Node("Skirt", parent=female)
skirt_color1 = Node("brown", parent=skirt)
skirt_color2 = Node("red", parent=skirt)
skirt_size1 = Node("small", parent=skirt)
blouse = Node("Blouse", parent=female)
blouse_color1 = Node("white", parent=blouse)
blouse_size1 = Node("small", parent=blouse)
blouse_size2 = Node("medium", parent=blouse)

DotExporter(start).to_picture("udo.png")

print("picture made")


util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
print("loaded")



def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    (https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    multi = norm_a * norm_b
    print(dot_product)
    print(multi)
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


def distance2vote(d, a=3, b=5):
    sim = np.maximum(0, 1 - d ** 2 / 2)
    return np.exp(-d ** a) * sim ** b


# define your word list
words_list = []
for synset in wn.all_synsets():
    word = str(synset.lemmas()[0].name())
    print(word)
    words_list.append(word)

words_vectors = {w: ft.get_sentence_vector(w) for w in words_list}
all_similarities = []

similarity_rose = compare_word('size', words_vectors)
similarity_rose = {k: v for k, v in sorted(similarity_rose.items(), key=lambda item: item[1])}
all_similarities.append(similarity_rose)
print("1/4")

similarity_chamomile = compare_word('color', words_vectors)
similarity_chamomile = {k: v for k, v in sorted(similarity_chamomile.items(), key=lambda item: item[1])}
all_similarities.append(similarity_chamomile)
print("2/4")


all_votes = []

for similarity in all_similarities:
    votes = {}
    for lemma_name in list(similarity)[-75:]:
        distance = similarity[lemma_name]
        for hyper in wn.synsets(lemma_name)[0].hypernyms():
            for lemma in hyper.lemmas():
                if lemma.name() not in votes.keys():
                    votes[lemma.name()] = 0
                votes[lemma.name()] += distance2vote(distance)
                print(f"{str(lemma_name)} {str(hyper)}")
    all_votes.append({k: v for k, v in sorted(votes.items(), key=lambda item: item[1])})

frequent_votes = {}


for vote in all_votes:
    vote_count = len(vote.keys())
    for i in range(vote_count-1, vote_count - 16, -1):
        lemma_name = list(vote.keys())[i]
        votes_received = vote[lemma_name]
        if lemma_name not in list(frequent_votes.keys()):
            frequent_votes[lemma_name] = 0
        frequent_votes[lemma_name] += votes_received

print({k: v for k, v in sorted(frequent_votes.items(), key=lambda item: item[1])})


