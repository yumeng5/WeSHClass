import numpy as np
import os
np.random.seed(1234)
from spherecluster import SphericalKMeans, VonMisesFisherMixture, sample_vMF
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
import pickle
from time import time
from multiprocessing import Pool


def sample_mix_vMF(center, kappa, weight, num_doc):
    distrib_idx = np.random.choice(range(len(center)), num_doc, p=weight)
    samples = []
    for idx in distrib_idx:
        samples.append(sample_vMF(center[idx], kappa[idx], 1))
    samples = np.array(samples)
    samples = np.reshape(samples, (num_doc, -1))
    return samples


def seed_expansion(relevant_nodes, prob_sup_array, sz, vocab_dict, embedding_mat):
    vocab_sz = len(vocab_dict)
    for j, relevant_node in enumerate(relevant_nodes):
        word_class = relevant_node.keywords
        prob_sup_class = prob_sup_array[j]
        expanded_class = []
        seed_vec = np.zeros(vocab_sz)
        if len(word_class) < sz:
            for i, word in enumerate(word_class):
                seed_vec[vocab_dict[word]] = prob_sup_class[i]
            expanded = np.dot(embedding_mat.transpose(), seed_vec)
            expanded = np.dot(embedding_mat, expanded)
            word_expanded = sorted(range(len(expanded)), key=lambda k: expanded[k], reverse=True)
            for i in range(sz):
                expanded_class.append(word_expanded[i])
            relevant_node.expanded = np.array(expanded_class)
        else:
            relevant_node.expanded = np.array([vocab_dict[w] for w in word_class])


def label_expansion(relevant_nodes, write_path, vocabulary_inv, embedding_mat, manual_num=None, fitting='mix'):
    print("Retrieving top-t nearest words...")
    vocab_dict = {v: k for k, v in vocabulary_inv.items()}
    prob_sup_array = []
    current_szes = []
    all_class_keywords = []
    children_nodes = []
    for relevant_node in relevant_nodes:
        if relevant_node.children:
            children_nodes += relevant_node.children
        else:
            children_nodes += [relevant_node]
    for children_node in children_nodes:
        current_sz = len(children_node.keywords)
        current_szes.append(current_sz)
        prob_sup_array.append([1/current_sz] * current_sz)
        all_class_keywords += children_node.keywords
    current_sz = np.min(current_szes)
    if manual_num is None:
        while len(all_class_keywords) == len(set(all_class_keywords)):
            print(f'current_sz: {current_sz}')
            current_sz += 1
            # print(f'len_kw: {len(all_class_keywords)}')
            seed_expansion(children_nodes, prob_sup_array, current_sz, vocab_dict, embedding_mat)
            all_class_keywords = [w for relevant_node in children_nodes for w in relevant_node.expanded]
        seed_expansion(children_nodes, prob_sup_array, current_sz-1, vocab_dict, embedding_mat)
        # seed_expansion(children_nodes, prob_sup_array, current_sz, vocab_dict, embedding_mat)
    else:
        seed_expansion(children_nodes, prob_sup_array, manual_num, vocab_dict, embedding_mat)
    if manual_num is None:
        print(f"Final expansion size t = {len(children_nodes[0].expanded)}")
    else:
        print(f"Manual expansion size t = {manual_num}")
    
    centers = []
    kappas = []
    weights = []
    if write_path is not None:
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        else:
            f = open(os.path.join(write_path, 'expanded.txt'), 'w')
            f.close()
    for relevant_node in relevant_nodes:
        children_nodes = relevant_node.children if relevant_node.children else [relevant_node]
        num_children = len(children_nodes)
        expanded_class = []
        if fitting == 'mix':
            for child in children_nodes:
                assert child.expanded != []
                expanded_class = np.concatenate((expanded_class, child.expanded))
                print([vocabulary_inv[w] for w in child.expanded])
            vocab_expanded = [vocabulary_inv[w] for w in expanded_class]
            expanded_mat = embedding_mat[np.asarray(list(set(expanded_class)), dtype='int32')]
            vmf_soft = VonMisesFisherMixture(n_clusters=num_children, n_jobs=15, random_state=0)
            vmf_soft.fit(expanded_mat)
            center = vmf_soft.cluster_centers_
            kappa = vmf_soft.concentrations_
            weight = vmf_soft.weights_
            print(f'weight: {weight}')
            print(f'kappa: {kappa}')
            centers.append(center)
            kappas.append(kappa)
            weights.append(weight)
        elif fitting == 'separate':
            center = []
            kappa = []
            weight = []
            for child in children_nodes:
                assert child.expanded != []
                expanded_class = np.concatenate((expanded_class, child.expanded))
                expanded_mat = embedding_mat[np.asarray(child.expanded, dtype='int32')]
                vmf_soft = VonMisesFisherMixture(n_clusters=1, n_jobs=15, random_state=0)
                vmf_soft.fit(expanded_mat)
                center.append(vmf_soft.cluster_centers_[0])
                kappa.append(vmf_soft.concentrations_[0])
                weight.append(1/num_children)
                expanded = np.dot(embedding_mat, center[-1])
                word_expanded = sorted(range(len(expanded)), key=lambda k: expanded[k], reverse=True)
            vocab_expanded = [vocabulary_inv[w] for w in expanded_class]
            print(f'Class {relevant_node.name}:')
            print(vocab_expanded)
            print(f'weight: {weight}')
            print(f'kappa: {kappa}')
            centers.append(center)
            kappas.append(kappa)
            weights.append(weight)
        if write_path is not None:
            f = open(os.path.join(write_path, 'expanded.txt'), 'a')
            f.write(relevant_node.name + '\t')
            f.write(' '.join(vocab_expanded) + '\n')
            f.close()
    
    print("Finished vMF distribution fitting.")
    return centers, kappas, weights


def bow_pseudodocs(relevant_nodes, expand_num, background_array, sequence_length, len_avg,
                    len_std, num_doc, interp_weight, vocabulary_inv, embedding_mat, save_dir=None, total_num=50):
    n_classes = len(relevant_nodes)

    # if os.path.exists(os.path.join(save_dir, 'pseudo_docs.pkl')):
    #     print(f'Loading pseudodocs for bow...')
    #     f = open(os.path.join(save_dir, 'pseudo_docs.pkl'), 'rb')
    #     docs, labels = pickle.load(f)
    #     f = open(os.path.join(save_dir, 'pseudo_docs.txt'), 'w')
    #     for doc in docs:
    #         f.write(" ".join([vocabulary_inv[ele] for ele in doc]) + '\n')
    #     f.close()
    #     return docs, labels

    for i in range(len(embedding_mat)):
        embedding_mat[i] = embedding_mat[i] / np.linalg.norm(embedding_mat[i])

    centers, kappas, weights = label_expansion(relevant_nodes, save_dir, vocabulary_inv, embedding_mat, expand_num)

    background_vec = interp_weight * background_array
    docs = np.zeros((num_doc*n_classes, sequence_length), dtype='int32')
    label = np.zeros((num_doc*n_classes, n_classes))
    
    for i in range(n_classes):
        docs_len = len_avg*np.ones(num_doc)
        center = centers[i]
        kappa = kappas[i]
        weight = weights[i]
        discourses = sample_mix_vMF(center, kappa, weight, num_doc)
        for j in range(num_doc):
            discourse = discourses[j]
            prob_vec = np.dot(embedding_mat, discourse)
            prob_vec = np.exp(prob_vec)
            sorted_idx = np.argsort(-prob_vec)
            delete_idx = sorted_idx[total_num:]
            prob_vec[delete_idx] = 0
            prob_vec /= np.sum(prob_vec)
            prob_vec *= 1 - interp_weight
            prob_vec += background_vec
            doc_len = int(docs_len[j])
            docs[i*num_doc+j][:doc_len] = np.random.choice(len(prob_vec), size=doc_len, p=prob_vec)
            label[i*num_doc+j] = interp_weight/n_classes*np.ones(n_classes)
            label[i*num_doc+j][i] += 1 - interp_weight

    f = open(os.path.join(save_dir, 'pseudo_docs_bow.txt'), 'w')
    for doc in docs:
        f.write(" ".join([vocabulary_inv[ele] for ele in doc]) + '\n')
    f.close()
    with open(os.path.join(save_dir, 'pseudo_docs_bow.pkl'), 'wb') as f:
        pickle.dump([docs, label], f, protocol=4)
    return docs, label


def lstm_pseudodocs(parent_node, expand_num, sequence_length, len_avg, sent_length, len_std, num_doc, 
                    interp_weight, vocabulary_inv, lm, common_words, save_dir=None):
    relevant_nodes = parent_node.children
    embedding_mat = parent_node.embedding
    n_classes = len(relevant_nodes)

    for i in range(len(embedding_mat)):
        embedding_mat[i] = embedding_mat[i] / np.linalg.norm(embedding_mat[i])

    centers, kappas, weights = label_expansion(relevant_nodes, save_dir, vocabulary_inv, embedding_mat, expand_num)

    seed_words = []
    for i in range(n_classes):
        center = centers[i]
        kappa = kappas[i]
        weight = weights[i]
        # discourses = sample_mix_vMF(center, kappa, weight, num_doc*num_sent)
        discourses = sample_mix_vMF(center, kappa, weight, num_doc)
        prob_mat = np.dot(discourses, embedding_mat.transpose())
        seeds = np.argmax(prob_mat, axis=1)
        seed_words.append(seeds)
        
    doc_len = int(len_avg)
    num_sent = int(np.ceil(doc_len/sent_length))
    docs = np.zeros((num_doc*n_classes, sequence_length), dtype='int32')
    label = np.zeros((num_doc*n_classes, n_classes))
    for i in range(n_classes):
        # seeds = np.reshape(seeds, (num_doc, num_sent))
        docs_class = gen_with_seeds(relevant_nodes[i].name, lm, seed_words[i], doc_len, sent_length, \
                                    common_words, vocabulary_inv, save_dir=save_dir)
        for j in range(num_doc):
            docs[i*num_doc+j, :doc_len] = docs_class[j]
            label[i*num_doc+j] = interp_weight/n_classes*np.ones(n_classes)
            label[i*num_doc+j][i] += 1 - interp_weight

    return docs, label


def gen_next(common_words, total_words, pred):
    select = np.random.choice(common_words+1, p=pred)
    pred_trim = select
    if select == common_words:
        pred_real = np.random.choice(range(common_words,total_words))
    else:
        pred_real = select
    return pred_real, pred_trim


def gen_with_seeds(class_name, lm, seeds, doc_len, sent_length, common_words, vocabulary_inv, save_dir=None):
    docs = np.zeros((len(seeds), doc_len), dtype='int32')
    # if os.path.exists(os.path.join(save_dir, f'{class_name}_pseudo_docs.pkl')):
    #     print(f'Loading pseudodocs for class {class_name}...')
    #     f = open(os.path.join(save_dir, f'{class_name}_pseudo_docs.pkl'), 'rb')
    #     return pickle.load(f)
    t0 = time()
    pool = Pool(10)
    doc_len = int(doc_len)
    
    sent_cnt = 0
    print(f'Pseudodocs generation for class {class_name}...')

    cur_seq = [[] for _ in range(len(seeds))]
    for i in range(doc_len):
        if i % sent_length == 0:
            # pred_real = [seed[sent_cnt] for seed in seeds]
            # pred_trim = [min(seed[sent_cnt], common_words) for seed in seeds]
            pred_real = [seed for seed in seeds]
            pred_trim = [min(seed, common_words) for seed in seeds]
            temp_seq = [[] for _ in range(len(seeds))]
            sent_cnt += 1
        else:
            padded_seq = pad_sequences(temp_seq, maxlen=sent_length-1, padding='pre')
            pred = lm.predict(padded_seq, verbose=0)
            args = [(common_words, len(vocabulary_inv), ele) for ele in pred]
            res = pool.starmap(gen_next, args)
            pred_real = [ele[0] for ele in res]
            pred_trim = [ele[1] for ele in res]
            assert len(pred_real) == len(cur_seq)
        for j in range(len(cur_seq)):
            cur_seq[j].append(pred_real[j])
            temp_seq[j].append(pred_trim[j])

    cur_seq = np.array(cur_seq)
    print(f'Pseudodocs generation time: {time() - t0:.2f}s')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, f'{class_name}_pseudo_docs.txt'), 'w')
    for seq in cur_seq:
        f.write(" ".join([vocabulary_inv[ele] for ele in seq]) + '\n')
    f.close()
    with open(os.path.join(save_dir, f'{class_name}_pseudo_docs.pkl'), 'wb') as f:
        pickle.dump(cur_seq, f, protocol=4)
    return cur_seq


def augment(x, relevant_nodes, total_len, save_dir=None):
    docs = []
    print("Labeled documents augmentation...")
    y = np.zeros((0, len(relevant_nodes)))
    sup_idx = []
    for i, node in enumerate(relevant_nodes):
        sup_idx += node.sup_idx
        labels = np.zeros((len(node.sup_idx), len(relevant_nodes)))
        labels[:, i] = 1.0
        y = np.concatenate((y, labels), axis=0)
    docs = x[sup_idx]
    curr_len = len(docs)
    copy_times = int(total_len/curr_len) - 1
    new_docs = docs
    new_y = y
    for _ in range(copy_times):
        new_docs = np.concatenate((new_docs, docs), axis=0)
        new_y = np.concatenate((new_y, y), axis=0)

    print("Finished labeled documents augmentation.")
    return new_docs, new_y
