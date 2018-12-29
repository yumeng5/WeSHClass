import numpy as np
np.random.seed(1234)
import os
from gensim.models import word2vec
from gen import augment, bow_pseudodocs, lstm_pseudodocs
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import f1_score
from time import time


def train_lstm(sequences, vocab_sz, truncate_len, save_path, word_embedding_dim=100, hidden_dim=100, embedding_matrix=None):
    if embedding_matrix is not None:
        trim_embedding = np.zeros((vocab_sz+1, word_embedding_dim))
        trim_embedding[:-1,:] = embedding_matrix[:vocab_sz,:]
        trim_embedding[-1,:] = np.average(embedding_matrix[vocab_sz:,:], axis=0)
    else:
        trim_embedding = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    from models import LSTMLanguageModel
    model_name = save_path + '/model-final.h5'
    model = LSTMLanguageModel(truncate_len-1, word_embedding_dim, vocab_sz+1, hidden_dim, trim_embedding)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists(model_name):
        print(f'Loading model {model_name}...')
        model.load_weights(model_name)
        return model
    x, y = sequences[:,:-1], sequences[:,-1]
    checkpointer = ModelCheckpoint(filepath=save_path + '/model-{epoch:02d}.h5', save_weights_only=True, period=1)
    model.fit(x, y, batch_size=256, epochs=25, verbose=1, callbacks=[checkpointer])
    model.save_weights(model_name)
    return model


def train_word2vec(sentence_matrix, vocabulary_inv, dataset_name, suffix='', mode='skipgram',
                   num_features=100, min_word_count=5, context=5, embedding_train=None):
    model_dir = './' + dataset_name
    model_name = 'embedding_' + suffix + '.p'
    model_name = os.path.join(model_dir, model_name)
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3
    print('Training Word2Vec model...')

    sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
    if mode == 'skipgram':
        sg = 1
        print('Model: skip-gram')
    elif mode == 'cbow':
        sg = 0
        print('Model: CBOW')
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, sg=sg,
                                        size=num_features, min_count=min_word_count,
                                        window=context, sample=downsampling)

    embedding_model.init_sims(replace=True)
    
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                        np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"Saving Word2Vec weights to {model_name}")
    pickle.dump(embedding_weights, open(model_name, "wb"))
    return embedding_weights


def train_class_embedding(x, vocabulary_inv, dataset_name, node, suffix='', mode='skipgram',
                         num_features=100, min_word_count=5, context=5):
    print(f'Training embedding for node {node.name}')
    
    model_dir = './' + dataset_name
    model_name = 'embedding_' + node.name + suffix + '.p'
    model_name = os.path.join(model_dir, model_name)
    if os.path.exists(model_name):
        print(f"Loading existing Word2Vec embedding {model_name}...")
        embedding_weights = pickle.load(open(model_name, "rb"))
        assert len(vocabulary_inv) == len(embedding_weights), f"Old word embedding model! Please delete {model_name} and re-run!"
    else:
        suffix = node.name + suffix
        embedding_weights = train_word2vec(x, vocabulary_inv, dataset_name, suffix, mode,
                                           num_features, min_word_count, context)
    embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocabulary_inv])
    node.embedding = embedding_mat


def proceed_level(x, sequences, wstc, args, pretrain_epochs, self_lr, decay, update_interval,
                delta, class_tree, level, expand_num, background_array, doc_length, sent_length, len_avg,
                len_std, num_doc, interp_weight, vocabulary_inv, common_words):
    print(f"\n### Proceeding level {level} ###")
    dataset = args.dataset
    sup_source = args.sup_source
    maxiter = args.maxiter.split(',')
    maxiter = int(maxiter[level])
    batch_size = args.batch_size
    parents = class_tree.find_at_level(level)
    parents_names = [parent.name for parent in parents]
    print(f'Nodes: {parents_names}')
    
    for parent in parents:
        # initialize classifiers in hierarchy
        print("\n### Input preparation ###")

        if class_tree.embedding is None:
            train_class_embedding(x, vocabulary_inv, dataset_name=args.dataset, node=class_tree)
        parent.embedding = class_tree.embedding
        wstc.instantiate(class_tree=parent)
        
        save_dir = f'./results/{dataset}/{sup_source}/level_{level}'

        if parent.model is not None:
            
            print("\n### Phase 1: vMF distribution fitting & pseudo document generation ###")

            if args.pseudo == "bow":
                print("Pseudo documents generation (Method: Bag-of-words)...")
                seed_docs, seed_label = bow_pseudodocs(parent.children, expand_num, background_array, doc_length, len_avg,
                                                        len_std, num_doc, interp_weight, vocabulary_inv, parent.embedding, save_dir)
            elif args.pseudo == "lstm":
                print("Pseudo documents generation (Method: LSTM language model)...")
                lm = train_lstm(sequences, common_words, sent_length, f'./{dataset}/lm', embedding_matrix=class_tree.embedding)
                
                seed_docs, seed_label = lstm_pseudodocs(parent, expand_num, doc_length, len_avg, sent_length, len_std, num_doc, 
                                                        interp_weight, vocabulary_inv, lm, common_words, save_dir)
            
            print("Finished pseudo documents generation.")
            num_real_doc = len(seed_docs) / 5

            if sup_source == 'docs':
                real_seed_docs, real_seed_label = augment(x, parent.children, num_real_doc)
                print(f'Labeled docs {len(real_seed_docs)} + Pseudo docs {len(seed_docs)}')
                seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
                seed_label = np.concatenate((seed_label, real_seed_label), axis=0)

            perm = np.random.permutation(len(seed_label))
            seed_docs = seed_docs[perm]
            seed_label = seed_label[perm]

            print('\n### Phase 2: pre-training with pseudo documents ###')
            print(f'Pretraining node {parent.name}')

            wstc.pretrain(x=seed_docs, pretrain_labels=seed_label, model=parent.model,
                        optimizer=SGD(lr=0.1, momentum=0.9),
                        epochs=pretrain_epochs, batch_size=batch_size,
                        save_dir=save_dir, suffix=parent.name)

    global_classifier = wstc.ensemble_classifier(level)
    wstc.model.append(global_classifier)
    t0 = time()
    print("\n### Phase 3: self-training ###")
    selftrain_optimizer = SGD(lr=self_lr, momentum=0.9, decay=decay)
    wstc.compile(level, optimizer=selftrain_optimizer, loss='kld')
    y_pred = wstc.fit(x, level=level, tol=delta, maxiter=maxiter, batch_size=batch_size,
                      update_interval=update_interval, save_dir=save_dir)
    print(f'Self-training time: {time() - t0:.2f}s')
    return y_pred


def f1(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_macro, f1_micro


def write_output(y_pred, perm, class_tree, write_path):
    invperm = np.zeros(len(perm), dtype='int32')
    for i,v in enumerate(perm):
        invperm[v] = i
    y_pred = y_pred[invperm]
    label2name = {}
    for i in range(class_tree.get_size()-1):
        label2name[i] = class_tree.find(i).name
    with open(os.path.join(write_path, 'out.txt'), 'w') as f:
        for val in y_pred:
            labels = np.nonzero(val)[0]
            if len(labels) > 0:
                out_str = '\t'.join([label2name[label] for label in labels])
            else:
                out_str = class_tree.name
            f.write(out_str + '\n')
    print("Classification results are written in {}".format(os.path.join(write_path, 'out.txt')))
