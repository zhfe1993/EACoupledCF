# coding=UTF-8
import gc
import time
from time import time
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape
from keras.layers import Embedding, Input, Multiply, Conv2D, Concatenate

from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from load_movie_data_cnn import load_itemGenres_as_matrix
from load_movie_data_cnn import load_negative_file
from load_movie_data_cnn import load_rating_file_as_list
from load_movie_data_cnn import load_rating_train_as_matrix
from load_movie_data_cnn import load_user_attributes
from evaluate_movie_cnn import evaluate_model

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


def get_model_0(num_users, num_items):
    # only global coupling
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(30,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_attr_input = Input(shape=(18,), dtype='float32', name='item_attr_input')
    item_attr_embedding = Dense(8, activation='relu')(item_attr_input)
    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])
    merge_attr_embedding_global = Flatten()(merge_attr_embedding)
    attr_1 = Dense(16)(merge_attr_embedding_global)
    attr_1 = Activation('relu')(attr_1)

    # ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])
    id_1 = Dense(64)(merge_id_embedding)
    id_1 = Activation('relu')(id_1)

    merge_attr_id_embedding = Concatenate()([attr_1, id_1])
    dense_1 = Dense(64)(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)

    topLayer = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(inputs=[user_attr_input, item_attr_input, user_id_input, item_id_input],
                  outputs=topLayer)

    return model


def get_model_1(num_users, num_items):
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(30,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_attr_input = Input(shape=(18,), dtype='float32', name='item_attr_input')
    item_attr_embedding = Dense(8, activation='relu')(item_attr_input)
    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding = Reshape((8, 8, 1))(merge_attr_embedding)
    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten()(merge_attr_embedding)

    attr_1 = Dense(16)(merge_attr_embedding)
    attr_1 = Activation('relu')(attr_1)

    ########################   id side   ##################################
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])

    # merge attr_id embedding
    merge_attr_id_embedding = Concatenate()([attr_1, merge_id_embedding])
    dense_1 = Dense(64)(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)

    topLayer = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(inputs=[user_attr_input, item_attr_input, user_id_input, item_id_input],
                  outputs=topLayer)

    return model


def get_model_2(num_users, num_items):
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################
    # Input
    user_attr_input = Input(shape=(30,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_attr_input = Input(shape=(18,), dtype='float32', name='item_attr_input')
    item_attr_embedding = Dense(8, activation='relu')(item_attr_input)
    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten()(merge_attr_embedding)

    merge_attr_embedding = Reshape((8, 8, 1))(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten()(merge_attr_embedding)
    merge_attr_embedding = Concatenate()([merge_attr_embedding, merge_attr_embedding_global])

    attr_1 = Dense(16)(merge_attr_embedding)
    attr_1 = Activation('relu')(attr_1)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])

    id_2 = Dense(32)(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = Concatenate()([attr_1, id_2])
    dense_1 = Dense(64)(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)

    topLayer = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(inputs=[user_attr_input, item_attr_input, user_id_input, item_id_input],
                  outputs=topLayer)

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
    model = get_model_0(num_users, num_items)

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
        hist = model.fit([user_attr_input, item_attr_input, user_id_input, item_id_input],
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
                                                 users_attr_mat, items_genres_mat, topK, evaluation_threads)
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
