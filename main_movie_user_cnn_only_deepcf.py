# coding=UTF-8
import gc
import time
from time import time

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten
from keras.layers import Embedding, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import tensorflow as tf
from load_movie_data_cnn import load_itemGenres_as_matrix
from load_movie_data_cnn import load_negative_file
from load_movie_data_cnn import load_rating_file_as_list
from load_movie_data_cnn import load_rating_train_as_matrix
from load_movie_data_cnn import load_user_attributes
from evaluate_movie_cnn_only_deepcf import evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically
sess = tf.Session(config=config)


def get_train_instances(users_attr_mat, ratings, items_genres_mat):
    user_attr_input, item_attr_input, user_id_input, item_id_input, labels = [], [], [], [], []
    num_users, num_items = ratings.shape
    num_negatives = 4

    for (u, i) in ratings.keys():
        # positive instance
        # user_vec_input.append(users_vec_mat[u])
        user_attr_input.append(users_attr_mat[u])
        user_id_input.append([u])
        item_id_input.append([i])
        item_attr_input.append(items_genres_mat[i])
        labels.append([1])

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)

            # user_vec_input.append(users_vec_mat[u])
            user_attr_input.append(users_attr_mat[u])
            user_id_input.append([u])
            item_id_input.append([j])
            item_attr_input.append(items_genres_mat[j])
            labels.append([0])
    # array_user_vec_input = np.array(user_vec_input)
    array_user_attr_input = np.array(user_attr_input)
    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_item_attr_input = np.array(item_attr_input)
    array_labels = np.array(labels)

    del user_attr_input, user_id_input, item_id_input, item_attr_input, labels
    gc.collect()

    return array_user_attr_input, array_user_id_input, array_item_attr_input, array_item_id_input, array_labels

def get_model_3(num_users, num_items):
    # only global coupling
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)

    id_2 = Dense(32)(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    # merge_attr_id_embedding = merge([attr_1, id_2], mode='concat')
    dense_1 = Dense(64)(id_2)
    dense_1 = Activation('relu')(dense_1)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_id_input, item_id_input],
                  output=topLayer)

    return model


def main():
    learning_rate = 0.001
    num_epochs = 50
    verbose = 1
    topK = 10
    evaluation_threads = 1
    num_negatives = 4
    startTime = time()

    # load data
    num_users, users_attr_mat = load_user_attributes()
    num_items, items_genres_mat = load_itemGenres_as_matrix()
    ratings = load_rating_train_as_matrix()

    # load model
    model = get_model_3(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    model.summary()

    # Training model
    best_hr, best_ndcg, best_mrr = 0, 0, 0
    for epoch in range(num_epochs):
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_attr_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(users_attr_mat,
                                                                                                     ratings,
                                                                                                     items_genres_mat)
        hist = model.fit([user_id_input, item_id_input],
                         labels, epochs=1,
                         batch_size=256,
                         verbose=2,
                         shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            testRatings = load_rating_file_as_list()
            testNegatives = load_negative_file()
            (hits, ndcgs, mrrs) = evaluate_model(model, testRatings, testNegatives,
                                                 topK, evaluation_threads)
            hr, ndcg, mrr, loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(mrrs).mean(), \
                                  hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, MRR = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, mrr, loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            if mrr > best_mrr:
                best_mrr = mrr
        endTime = time()
        print("End. best HR = %.4f, best NDCG = %.4f, best_MRR = %.4f, time = %.1f s" %
              (best_hr, best_ndcg, best_mrr, endTime - startTime))
        print('HR = %.4f, NDCG = %.4f, MRR = %.4f' % (hr, ndcg, mrr))


if __name__ == '__main__':
    main()
