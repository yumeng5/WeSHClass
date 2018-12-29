import csv
import numpy as np
import os
import re
import itertools
import pickle
import nltk
from collections import Counter
from os.path import join
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from tree import ClassNode


def read_file(dataset, with_eval="All"):
    class_tree = ClassNode("ROOT",None,-1)
    hier_file = open(f"./{dataset}/label_hier.txt", 'r')
    contents = hier_file.readlines()
    cnt = 0
    for line in contents:
        line = line.split("\n")[0]
        line = line.split("\t")
        parent = line[0]
        children = line[1:]
        for child in children:
            parent_node = class_tree.find(parent)
            class_tree.find_add_child(parent, ClassNode(child, parent_node))
            cnt += 1
    
    # assign labels to classes in class tree
    offset = 0
    for i in range(1, class_tree.get_height()+1):
        nodes = class_tree.find_at_level(i)
        for node in nodes:
            node.label = offset
            offset += 1

    n_classes = class_tree.get_size() - 1
    print(f'Total number of classes: {n_classes}')
    print(class_tree.visualize_tree())
    
    infile = open(f'./{dataset}/dataset.txt', mode='r', encoding='utf-8')
    data = infile.readlines()
    if with_eval == "All":
        infile = open(f'./{dataset}/labels.txt', mode='r', encoding='utf-8')
        labels = infile.readlines()
        y = []
        for line in labels:
            label = line.split('\n')[0]
            current = np.zeros(n_classes)
            for i in class_tree.name2label(label):
                current[i] = 1.0
            y.append(current)
        y = np.asarray(y)
        assert len(data) == len(y)
    elif with_eval == "Subset":
        infile = open(f'./{dataset}/labels_sub.txt', mode='r', encoding='utf-8')
        labels = infile.readlines()
        y = {}
        for line in labels:
            label = line.split('\n')[0]
            class_name = label.split('\t')[0]
            doc_ids = label.split('\t')[1]
            doc_ids = [int(doc_id) for doc_id in doc_ids.split(' ')]
            for doc_id in doc_ids:
                current = np.zeros(n_classes)
                for i in class_tree.name2label(class_name):
                    current[i] = 1.0
                y[doc_id] = current
    else:
        y = None
    return data, y, class_tree


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\'-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_doc(data):
    data = [s.strip() for s in data]
    data = [clean_str(s) for s in data]
    return data


def pad_docs(sentences, pad_len=None, padding_word="<PAD/>"):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for sentence in sentences:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, common_words):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    trim_vocabulary = {}
    for i, x in enumerate(vocabulary_inv):
        if i < common_words:
            trim_vocabulary[x] = i
        else:
            trim_vocabulary[x] = common_words
    return word_counts, vocabulary, vocabulary_inv, trim_vocabulary


def build_sequence(flat_data, vocabulary, truncate_len):
    flat_data = build_input_data(flat_data, vocabulary)
    sequences = []
    for seq in flat_data:
        for i in range(1, len(seq)):
            sequence = seq[:i+1]
            sequences.append(sequence)
    sequences = pad_sequences(sequences, maxlen=truncate_len, padding='pre')
    print(f'Sequences shape: {sequences.shape}')
    return sequences


def build_input_data(sentences, vocabulary):
    x = [[vocabulary[word] for word in sentence] for sentence in sentences]
    return x


def extract_keywords(data_path, class_tree, class_type, vocabulary, num_seed_doc, num_keywords, data, perm):
    data = [' '.join(line) for line in data]
    tfidf = TfidfVectorizer(norm='l2', sublinear_tf=True, max_df=0.2, stop_words='english', 
                            token_pattern=r'(?u)\b\w[\w-]*\w\b', max_features=10000)
    print("\n### Supervision type: Labeled documents ###")

    file_name = f'doc_id.txt'
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    for line in text:
        line = line.split('\n')[0]
        class_name, doc_ids = line.split('\t')
        cur_node = class_tree.find(class_name)
        assert cur_node, f"Class {class_name} not exist in class tree!"
        seed_idx = doc_ids.split()
        seed_idx = [int(idx) for idx in seed_idx]
        cur_node.sup_idx = seed_idx

    print("Extracted keywords for each class: ")
    max_level = class_tree.get_height()
    for level in reversed(range(1, max_level+1)):
        nodes = class_tree.find_at_level(level)
        all_idx = []
        for node in nodes:
            if node.children == []:
                assert node.sup_idx, f"{node.name} has no labeled documents!"
            node.parent.sup_idx += node.sup_idx
            all_idx += node.sup_idx
        data_level = [data[idx] for idx in all_idx]
        x_level = tfidf.fit_transform(data_level)
        vocab_dict = tfidf.vocabulary_
        vocab_inv_dict = {v: k for k, v in vocab_dict.items()}
        cum_cnt = 0
        for node in nodes:
            x_node = x_level[cum_cnt:cum_cnt + len(node.sup_idx)].todense()
            cum_cnt += len(node.sup_idx)
            class_vec = np.average(x_node, axis=0)
            class_vec = np.ravel(class_vec)
            sort_idx = np.argsort(class_vec)[::-1]
            keyword = []
            if class_type == 'topic':
                j = 0
                k = 0
                while j < num_keywords:
                    w = vocab_inv_dict[sort_idx[k]]
                    if w in vocabulary:
                        keyword.append(vocab_inv_dict[sort_idx[k]])
                        j += 1
                    k += 1
            elif class_type == 'sentiment':
                j = 0
                k = 0
                while j < num_keywords:
                    w = vocab_inv_dict[sort_idx[k]]
                    w, t = nltk.pos_tag([w])[0]
                    if t.startswith("J") and w in vocabulary:
                        keyword.append(w)
                        j += 1
                    k += 1
            print(f'{node.name}: {keyword}')
            node.add_keywords(keyword)


def load_keywords(data_path, class_tree):
    file_name = 'keywords.txt'
    print("\n### Supervision type: Class-related Keywords ###")
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    
    for line in text:
        line = line.split('\n')[0]
        class_name, keywords = line.split('\t')
        keywords = keywords.split()
        class_tree.find_add_keywords(class_name, keywords)
    
    class_tree.aggregate_keywords()


def load_dataset(dataset_name, sup_source, num_seed_doc=10, common_words=10000, truncate_doc_len=None, truncate_sent_len=None, with_eval=True):
    data_path = './' + dataset_name
    data, y, class_tree = read_file(dataset_name, with_eval=with_eval)

    np.random.seed(1234)

    data = preprocess_doc(data)
    data = [s.split(" ") for s in data]
    trun_data = [s[:truncate_doc_len] for s in data]
    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics - Documents: ###")
    print(f'Document max length: {len_max} (words)')
    print(f'Document average length: {len_avg} (words)')
    print(f'Document length std: {len_std} (words)')

    if truncate_doc_len is None:
        truncate_doc_len = min(int(len_avg + 3*len_std), len_max)
    print(f"Defined maximum document length: {truncate_doc_len} (words)")
    print(f'Fraction of truncated documents: {sum(tmp > truncate_doc_len for tmp in tmp_list)/len(tmp_list)}')
    
    sequences_padded = pad_docs(trun_data, pad_len=truncate_doc_len)
    word_counts, vocabulary, vocabulary_inv, trim_vocabulary = build_vocab(sequences_padded, common_words)
    print(f"Vocabulary Size: {len(vocabulary_inv):d}")
    x = build_input_data(sequences_padded, vocabulary)
    x = np.array(x)

    # Prepare sentences for training LSTM language model
    trun_data = [" ".join(doc) for doc in trun_data]
    flat_data = [tokenize.sent_tokenize(doc) for doc in trun_data]
    flat_data = [sent for doc in flat_data for sent in doc]
    flat_data = [sent for sent in flat_data if len(sent.split(" ")) > 5]
    tmp_list = [len(sent.split(" ")) for sent in flat_data]
    max_sent_len = max(tmp_list)
    avg_sent_len = np.average(tmp_list)
    std_sent_len = np.std(tmp_list)
    if truncate_sent_len is None:
        truncate_sent_len = min(int(avg_sent_len + 3*std_sent_len), max_sent_len)
    print("\n### Dataset statistics - Sentences: ###")
    print(f'Sentence max length: {max_sent_len} (words)')
    print(f'Sentence average length: {avg_sent_len} (words)')
    print(f"Defined maximum sentence length: {truncate_sent_len} (words)")
    print(f'Fraction of truncated sentences: {sum(tmp > truncate_sent_len for tmp in tmp_list)/len(tmp_list)}')
    flat_data = [s.split(" ") for s in flat_data]
    sequences = build_sequence(flat_data, trim_vocabulary, truncate_sent_len)

    perm = np.random.permutation(len(x))
    if sup_source == 'keywords':
        load_keywords(data_path, class_tree)
    elif sup_source == 'docs':
        if dataset_name == 'yelp':
            class_type = 'sentiment'
            num_keywords = 5
        else:
            class_type = 'topic'
            num_keywords = 10
        extract_keywords(data_path, class_tree, class_type, vocabulary, num_seed_doc, num_keywords, data, perm)
    x = x[perm]
    if y is not None:
        if type(y) == dict:
            inv_perm = {k: v for v, k in enumerate(perm)}
            perm_y = {}
            for doc_id in y:
                perm_y[inv_perm[doc_id]] = y[doc_id]
            y = perm_y
        else:
            y = y[perm]
    return x, y, sequences, class_tree, word_counts, vocabulary, vocabulary_inv, len_avg, len_std, perm
