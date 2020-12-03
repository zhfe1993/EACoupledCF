# -*- coding: utf-8 -*-
# @Time     : 2020/11/20 15:48
# @Author   : Zhangfeng
# @FileName : NeuMF.py
# @Mail     : 1198211355@qq.com


import gc
import time
from time import time

import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal,TruncatedNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input , Add,Multiply, Conv2D ,Dropout ,Concatenate

from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
import os
from LoadMovieDataCnn import load_itemGenres_as_matrix
from LoadMovieDataCnn import load_negative_file
from LoadMovieDataCnn import load_rating_file_as_list
from LoadMovieDataCnn import load_rating_train_as_matrix
from LoadMovieDataCnn import load_user_attributes
from evaluateNeuMF import evaluate_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_train_instances(ratings):
    user_id_input, item_id_input, labels = [],[],[]
    num_users = ratings.shape[0]
    num_items =  ratings.shape[1]
    num_negatives = 4
    for (u, i) in ratings.keys():
        # positive instance
        user_id_input.append(u)
        item_id_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)
            user_id_input.append(u)
            item_id_input.append(j)
            labels.append(0)

    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_labels = np.array(labels)

    del  user_id_input, item_id_input, labels
    gc.collect()
    return array_user_id_input, array_item_id_input, array_labels

#GMF
def get_model_0(num_users, num_items, latent_dim, regs=[0, 0]):
    # Input variables
    num_users = num_users + 1
    num_items = num_items + 1

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=8, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  kernel_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=8, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  kernel_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_id_Embedding, item_id_Embedding])


    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

    model = Model(inputs=[user_id_input, item_id_input],
                  outputs=prediction)

    return model

#MLP
def get_model_1(num_users, num_items):
    num_users = num_users + 1
    num_items = num_items + 1
    layers = [20, 10]
    reg_layers = [0,0]
    assert len(layers) == len(reg_layers)

    # Number of layers in the MLP
    num_layer = len(layers)
    num_users = num_users + 1
    num_items = num_items + 1
    # Input variables
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=layers[0]/2, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  kernel_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=layers[0]/2, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  kernel_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate()([user_id_Embedding, item_id_Embedding])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_id_input, item_id_input],
                  outputs=prediction)

    return model

# NeuMF
def get_model_2(num_users, num_items):
    num_users = num_users + 1
    num_items = num_items + 1
    layers = [20,10]
    reg_layers = [0]
    reg_mf = 0
    assert len(layers) == len(reg_layers)
    # Number of layers in the MLP
    num_layer = len(layers)
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    mf_user_id_Embedding = Embedding(input_dim=num_users, output_dim=8, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)
    mf_item_id_Embedding = Embedding(input_dim=num_items, output_dim=8, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)

    mlp_user_id_Embedding = Embedding(input_dim=num_users, output_dim=int(layers[0]/2), name='mlp_user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)

    mlp_item_id_Embedding = Embedding(input_dim=num_items, output_dim=int(layers[0]/2), name='mlp_item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None), input_length=1)

    mf_user_id_Embedding = Flatten()(mf_user_id_Embedding(user_id_input))
    mf_item_id_Embedding = Flatten()(mf_item_id_Embedding(item_id_input))

    mlp_user_id_Embedding = Flatten()(mlp_user_id_Embedding(user_id_input))
    mlp_item_id_Embedding = Flatten()(mlp_item_id_Embedding(item_id_input))

    mf_vector = Multiply()([mf_user_id_Embedding, mf_item_id_Embedding])
    mlp_vector = Concatenate()([mlp_user_id_Embedding, mlp_item_id_Embedding])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        mlp_vector = layer(mlp_vector)

    predict_vector  =  Concatenate()([mf_vector, mlp_vector])
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = Model(inputs=[user_id_input, item_id_input],
                  outputs=prediction)

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
    # users_vec_mat = load_user_vectors()
    ratings = load_rating_train_as_matrix()

    # load model
    model = get_model_2(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    # plot_model(model, show_shapes=True, to_file='mainMovieUserCnn.png')
    model.summary()

    # Training model
    best_hr, best_ndcg = 0, 0
    for epoch in range(num_epochs):
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_id_input, item_id_input, labels = get_train_instances( ratings)
        hist = model.fit([user_id_input, item_id_input,user_id_input, item_id_input],labels, epochs=1,
                         batch_size=256,
                         verbose=2,
                         shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            testRatings = load_rating_file_as_list()
            testNegatives = load_negative_file()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives,topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
#                if hr > 0.6:
#                    model.save_weights('Pretrain/movielens_1m_only_global_neg_%d_hr_%.4f_ndcg_%.4f.h5' %
#                                      (num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f,time = %.1f s" %
          (best_hr, best_ndcg, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))

if __name__ == '__main__':
    main()
