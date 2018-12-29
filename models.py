import numpy as np
np.random.seed(1234)
import os
from time import time
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import csv
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Convolution1D, Embedding, GlobalMaxPooling1D, LSTM, Multiply, Lambda, Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints
from keras.initializers import RandomUniform
from utils import f1
from scipy.stats import entropy


def LSTMLanguageModel(input_shape, word_embedding_dim, vocab_sz, hidden_dim, embedding_matrix):
    x = Input(shape=(input_shape,), name='input')
    z = Embedding(vocab_sz, word_embedding_dim, input_length=input_shape, weights=[embedding_matrix], trainable=False)(x)
    z = LSTM(hidden_dim, activation='relu', return_sequences=True)(z)
    z = LSTM(hidden_dim, activation='relu')(z)
    z = Dense(vocab_sz, activation='softmax')(z)
    model = Model(inputs=x, outputs=z)
    model.summary()
    return Model(inputs=x, outputs=z)


def ConvolutionLayer(x, input_shape, n_classes, filter_sizes=[2, 3, 4, 5], num_filters=20, word_trainable=False,
                     vocab_sz=None,
                     embedding_matrix=None, word_embedding_dim=100, hidden_dim=100, act='relu', init='ones'):
    if embedding_matrix is not None:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,),
                      weights=[embedding_matrix], trainable=word_trainable)(x)
    else:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,), trainable=word_trainable)(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation=act,
                             strides=1,
                             kernel_initializer=init)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=x, outputs=y)


def IndexLayer(idx):
    def func(x):
        return x[:, idx]

    return Lambda(func)


def ExpanLayer(dim):
    def func(x):
        return K.expand_dims(x, dim)

    return Lambda(func)


class WSTC(object):
    def __init__(self,
                 input_shape,
                 class_tree,
                 max_level,
                 sup_source,
                 init=RandomUniform(minval=-0.01, maxval=0.01),
                 y=None,
                 vocab_sz=None,
                 word_embedding_dim=100,
                 blocking_perc=0,
                 block_thre=1.0,
                 block_level=1,
                 ):

        super(WSTC, self).__init__()

        self.input_shape = input_shape
        self.class_tree = class_tree
        self.y = y
        if type(y) == dict:
            self.eval_set = np.array([ele for ele in y])
        else:
            self.eval_set = None
        self.vocab_sz = vocab_sz
        self.block_level = block_level
        self.block_thre = block_thre
        self.block_label = {}
        self.siblings_map = {}
        self.x = Input(shape=(input_shape[1],), name='input')
        self.model = []
        self.sup_dict = {}
        if sup_source == 'docs':
            n_classes = class_tree.get_size() - 1
            leaves = class_tree.find_leaves()
            for leaf in leaves:
                current = np.zeros(n_classes)
                for i in class_tree.name2label(leaf.name):
                    current[i] = 1.0
                for idx in leaf.sup_idx:
                    self.sup_dict[idx] = current

    def instantiate(self, class_tree, filter_sizes=[2, 3, 4, 5], num_filters=20, word_trainable=False,
                    word_embedding_dim=100, hidden_dim=20, act='relu', init=RandomUniform(minval=-0.01, maxval=0.01)):
        num_children = len(class_tree.children)
        if num_children <= 1:
            class_tree.model = None
        else:
            class_tree.model = ConvolutionLayer(self.x, self.input_shape[1], filter_sizes=filter_sizes,
                                                n_classes=num_children,
                                                vocab_sz=self.vocab_sz, embedding_matrix=class_tree.embedding,
                                                hidden_dim=hidden_dim,
                                                word_embedding_dim=word_embedding_dim, num_filters=num_filters,
                                                init=init,
                                                word_trainable=word_trainable, act=act)

    def ensemble(self, class_tree, level, input_shape, parent_output):
        outputs = []
        if class_tree.model:
            y_curr = class_tree.model(self.x)
            if parent_output is not None:
                y_curr = Multiply()([parent_output, y_curr])
        else:
            y_curr = parent_output

        if level == 0:
            outputs.append(y_curr)
        else:
            for i, child in enumerate(class_tree.children):
                outputs += self.ensemble(child, level - 1, input_shape, IndexLayer(i)(y_curr))
        return outputs

    def ensemble_classifier(self, level):
        outputs = self.ensemble(self.class_tree, level, self.input_shape[1], None)
        outputs = [ExpanLayer(-1)(output) if len(output.get_shape()) < 2 else output for output in outputs]
        z = Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
        return Model(inputs=self.x, outputs=z)

    def pretrain(self, x, pretrain_labels, model, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None, suffix=''):

        model.compile(optimizer=optimizer, loss=loss)
        t0 = time()
        print('\nPretraining...')
        model.fit(x, pretrain_labels, batch_size=batch_size, epochs=epochs)
        print(f'Pretraining time: {time() - t0:.2f}s')
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save_weights(f'{save_dir}/pretrained_{suffix}.h5')

    def load_weights(self, weights, level):
        print(f'Loading weights @ level {level}')
        self.model[level].load_weights(weights)

    def load_pretrain(self, weights, model):
        model.load_weights(weights)

    def extract_label(self, y, level):
        if type(level) is int:
            relevant_nodes = self.class_tree.find_at_level(level)
            relevant_labels = [relevant_node.label for relevant_node in relevant_nodes]
        else:
            relevant_labels = []
            for i in level:
                relevant_nodes = self.class_tree.find_at_level(i)
                relevant_labels += [relevant_node.label for relevant_node in relevant_nodes]
        if type(y) is dict:
            y_ret = {}
            for key in y:
                y_ret[key] = y[key][relevant_labels]
        else:
            y_ret = y[:, relevant_labels]
        return y_ret

    def predict(self, x, level):
        q = self.model[level].predict(x, verbose=0)
        return q.argmax(1)

    def expand_pred(self, q_pred, level, cur_idx):
        y_expanded = np.zeros((self.input_shape[0], q_pred.shape[1]))
        if level not in self.siblings_map:
            self.siblings_map[level] = self.class_tree.siblings_at_level(level)
        siblings_map = self.siblings_map[level]
        block_idx = []
        for i, q in enumerate(q_pred):
            pred = np.argmax(q)
            idx = cur_idx[i]
            if level >= self.block_level and self.block_thre < 1.0 and idx not in self.sup_dict:
                siblings = siblings_map[pred]
                siblings_pred = q[siblings]/np.sum(q[siblings])
                if len(siblings) >= 2:
                    conf_val = entropy(siblings_pred)/np.log(len(siblings))
                else:
                    conf_val = 0
                if conf_val > self.block_thre:
                    block_idx.append(idx)
                else:
                    y_expanded[idx,pred] = 1.0
            else:
                y_expanded[idx,pred] = 1.0
        if self.block_label:
            blocked = [idx for idx in self.block_label]
            blocked_labels = np.array([label for label in self.block_label.values()])
            blocked_labels = self.extract_label(blocked_labels, level+1)
            y_expanded[blocked,:] = blocked_labels
        return y_expanded, block_idx

    def aggregate_pred(self, q_all, level, block_idx, cur_idx, agg="All"):
        leaves = self.class_tree.find_at_level(level+1)
        leaves_labels = [leaf.label for leaf in leaves]
        parents = self.class_tree.find_at_level(level)
        parents_labels = [parent.label for parent in parents]
        ancestor_dict = {}
        for leaf in leaves:
            ancestors = leaf.find_ancestors()
            ancestor_dict[leaf.label] = [ancestor.label for ancestor in ancestors]
        for parent in parents:
            ancestors = parent.find_ancestors()
            ancestor_dict[parent.label] = [ancestor.label for ancestor in ancestors]
        y_leaf = np.argmax(q_all[:, leaves_labels], axis=1)
        y_leaf = [leaves_labels[y] for y in y_leaf]
        if level > 0:
            y_parents = np.argmax(q_all[:, parents_labels], axis=1)
            y_parents = [parents_labels[y] for y in y_parents]
        if agg == "Subset" and self.eval_set is not None:
            cur_eval = [ele for ele in self.eval_set if ele in cur_idx]
            inv_cur_idx = {i:idx for idx, i in enumerate(cur_idx)}
            y_aggregate = np.zeros((len(cur_eval), q_all.shape[1]))
            for i, raw_idx in enumerate(cur_eval):
                idx = inv_cur_idx[raw_idx]
                if raw_idx not in block_idx:
                    y_aggregate[i, y_leaf[idx]] = 1.0
                    for ancestor in ancestor_dict[y_leaf[idx]]:
                        y_aggregate[i, ancestor] = 1.0
                else:
                    if level > 0:
                        y_aggregate[i, y_parents[idx]] = 1.0
                        for ancestor in ancestor_dict[y_parents[idx]]:
                            y_aggregate[i, ancestor] = 1.0
        else:
            y_aggregate = np.zeros((self.input_shape[0], q_all.shape[1]))
            for i in range(len(q_all)):
                idx = cur_idx[i]
                if idx not in block_idx:
                    y_aggregate[idx, y_leaf[i]] = 1.0
                    for ancestor in ancestor_dict[y_leaf[i]]:
                        y_aggregate[idx, ancestor] = 1.0
                else:
                    if level > 0:
                        y_aggregate[idx, y_parents[i]] = 1.0
                        for ancestor in ancestor_dict[y_parents[i]]:
                            y_aggregate[idx, ancestor] = 1.0
            if self.block_label:
                blocked = [idx for idx in self.block_label]
                blocked_labels = np.array([label for label in self.block_label.values()])
                blocked_labels = self.extract_label(blocked_labels, range(1, level+2))
                y_aggregate[blocked, :] = blocked_labels
        return y_aggregate

    def record_block(self, block_idx, y_pred_agg):
        n_classes = self.class_tree.get_size() - 1
        for idx in block_idx:
            self.block_label[idx] = np.zeros(n_classes)
            self.block_label[idx][:len(y_pred_agg[idx])] = y_pred_agg[idx]

    def target_distribution(self, q, nonblock, sup_level, power=2):
        q = q[nonblock]
        weight = q ** power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        inv_nonblock = {k:v for v,k in enumerate(nonblock)}
        for i in sup_level:
            mapped_i = inv_nonblock[i]
            p[mapped_i] = sup_level[i]
        return p

    def compile(self, level, optimizer='sgd', loss='kld'):
        self.model[level].compile(optimizer=optimizer, loss=loss)
        # print(f"\nLevel {level} model summary: ")
        # self.model[level].summary()

    def fit(self, x, level, maxiter=5e4, batch_size=256, tol=0.1, power=2,
            update_interval=100, save_dir=None, save_suffix=''):
        model = self.model[level]
        print(f'Update interval: {update_interval}')
        
        cur_idx = np.array([idx for idx in range(x.shape[0]) if idx not in self.block_label])
        x = x[cur_idx]
        y = self.y

        # logging files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfiles = []
        logwriters = []
        for i in range(level+2):
            if i <= level:
                logfile = open(save_dir + f'/self_training_log_level_{i}{save_suffix}.csv', 'w')
            else:
                logfile = open(save_dir + f'/self_training_log_all{save_suffix}.csv', 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'f1_macro', 'f1_micro'])
            logwriter.writeheader()
            logfiles.append(logfile)
            logwriters.append(logwriter)

        index = 0

        if y is not None:
            if self.eval_set is not None:
                cur_eval = [idx for idx in self.eval_set if idx in cur_idx]
                y = np.array([y[idx] for idx in cur_eval])
            y_all = []
            label_all = []
            for i in range(level+1):
                y_curr = self.extract_label(y, i+1)
                y_all.append(y_curr)
                nodes = self.class_tree.find_at_level(i+1)
                label_all += [node.label for node in nodes]
            y = y[:, label_all]

        mapped_sup_dict_level = {}
        if len(self.sup_dict) > 0:
            sup_dict_level = self.extract_label(self.sup_dict, level+1)
            inv_cur_idx = {i:idx for idx, i in enumerate(cur_idx)}        
            for key in sup_dict_level:
                mapped_sup_dict_level[inv_cur_idx[key]] = sup_dict_level[key]
    
        for ite in range(int(maxiter)):
            try:
                if ite % update_interval == 0:
                    print(f'\nIter {ite}: ')
                    y_pred_all = []
                    q_all = np.zeros((len(x), 0))
                    for i in range(level+1):
                        q_i = self.model[i].predict(x)
                        q_all = np.concatenate((q_all, q_i), axis=1)
                        y_pred_i, block_idx = self.expand_pred(q_i, i, cur_idx)
                        y_pred_all.append(y_pred_i)
                    q = q_i
                    y_pred = y_pred_i
                    if len(block_idx) > 0:
                        print(f'Number of blocked documents back to level {level}: {len(block_idx)}')
                    y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)

                    if y is not None:
                        if self.eval_set is not None:
                            y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx, agg="Subset")
                            y_pred_all = [y_pred[cur_eval, :] for y_pred in y_pred_all]
                            for i in range(level+1):
                                f1_macro, f1_micro = np.round(f1(y_all[i], y_pred_all[i]), 5)
                                print(f'Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}')
                                logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                                logwriters[i].writerow(logdict)
                            f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
                            logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                            logwriters[-1].writerow(logdict)
                            print(f'Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes')
                        else:
                            y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)
                            for i in range(level+1):
                                f1_macro, f1_micro = np.round(f1(y_all[i], y_pred_all[i]), 5)
                                print(f'f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}')
                                logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                                logwriters[i].writerow(logdict)
                            f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
                            logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                            logwriters[-1].writerow(logdict)
                            print(f'f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes')
                        
                    nonblock = np.array(list(set(range(x.shape[0])) - set(block_idx)))
                    x_nonblock = x[nonblock]
                    p_nonblock = self.target_distribution(q, nonblock, mapped_sup_dict_level, power)

                    if ite > 0:
                        change_idx = []
                        for i in range(len(y_pred)):
                            if not np.array_equal(y_pred[i], y_pred_last[i]):
                                change_idx.append(i)
                        y_pred_last = np.copy(y_pred)
                        delta_label = len(change_idx)
                        print(f'Fraction of documents with label changes: {np.round(delta_label/y_pred.shape[0]*100, 3)} %')
                        
                        if delta_label/y_pred.shape[0] < tol/100:
                            print(f'\nFraction: {np.round(delta_label / y_pred.shape[0] * 100, 3)} % < tol: {tol} %')
                            print('Reached tolerance threshold. Self-training terminated.')
                            break
                    else:
                        y_pred_last = np.copy(y_pred)

                # train on batch
                index_array = np.arange(x_nonblock.shape[0])
                if index * batch_size >= x_nonblock.shape[0]:
                    index = 0
                idx = index_array[index * batch_size: min((index + 1) * batch_size, x_nonblock.shape[0])]
                try:
                    assert len(idx) > 0
                except AssertionError:
                    print(f'Error @ index {index}')
                model.train_on_batch(x=x_nonblock[idx], y=p_nonblock[idx])
                index = index + 1 if (index + 1) * batch_size < x_nonblock.shape[0] else 0
                ite += 1

            except KeyboardInterrupt:
                print("\nKeyboard interrupt! Self-training terminated.")
                break

        for logfile in logfiles:
            logfile.close()

        if save_dir is not None:
            model.save_weights(save_dir + '/final.h5')
            print(f"Final model saved to: {save_dir}/final.h5")
        q_all = np.zeros((len(x), 0))
        for i in range(level+1):
            q_i = self.model[i].predict(x)
            q_all = np.concatenate((q_all, q_i), axis=1)
        y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)
        self.record_block(block_idx, y_pred_agg)
        return y_pred_agg
