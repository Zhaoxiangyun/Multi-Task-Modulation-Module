"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import facenet
import cv2
import random
import pdb
import h5py
import math


def main(args):
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + args.experiment_name
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    class_name = ['smile','oval_face','5_ocloc_shadow','bald','archied_eyebrows','Big_lips', 'Big_Nose']
    class_num = len(class_name)
    class_index = [31,25,0,4,1,6,7]
    all_image_list = []
    all_label_list = []
    for i in range(class_num):

        image_list = []
        label_list = []

        train_set = facenet.get_sub_category_dataset(args.data_dir, class_index[i])

        image_list_p, label_list_p = facenet.get_image_paths_and_labels_triplet(train_set[0], args)
        image_list_n, label_list_n = facenet.get_image_paths_and_labels_triplet(train_set[1], args)

        image_list.append(image_list_p)
        image_list.append(image_list_n)
        label_list.append(label_list_p)
        label_list.append(label_list_n)

        all_image_list.append(image_list)
        all_label_list.append(label_list)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
        
    image_size = args.image_size
    batch_size = args.batch_size
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')


        # image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        image_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size,3), name='images')
        labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size,3), name='labels')

        code_placeholder = tf.placeholder(tf.float32, shape=(batch_size,class_num,1,1), name='code')

        image_batch = normalized_image(image_placeholder)
        code_batch = code_placeholder

        control_code = tf.tile(code_placeholder,[1,1,args.embedding_size,1])
        mask_array = np.ones((1 ,class_num,args.embedding_size,1),np.float32)

        # for i in range(class_num):
        #     mask_array[:,i,(args.embedding_size/class_num)*i:(args.embedding_size/class_num)*(i+1)] = 1


        mask_tensor = tf.get_variable('mask', dtype=tf.float32, trainable=args.learned_mask, initializer=tf.constant(mask_array))
        mask_tensor = tf.tile(mask_tensor,[batch_size,1,1,1])
        control_code = tf.tile(code_placeholder,[1,1,args.embedding_size,1])

        mask_out = tf.multiply(mask_tensor, control_code)
        mask_out = tf.reduce_sum(mask_out,axis=1)
        mask_out = tf.squeeze(mask_out)
        mask_out = tf.nn.relu(mask_out)

        mask0_array = np.ones((1, class_num, 128, 1), np.float32)
        mask0_tensor = tf.get_variable('mask0', dtype=tf.float32, trainable=args.learned_mask,
                                      initializer=tf.constant(mask0_array))
        mask0_tensor = tf.tile(mask0_tensor, [batch_size, 1, 1, 1])
        control0_code = tf.tile(code_placeholder,[1,1,128,1])

        mask0_out = tf.multiply(mask0_tensor, control0_code)
        mask0_out = tf.reduce_sum(mask0_out, axis=1)
        mask0_out = tf.squeeze(mask0_out)
        mask0_out = tf.nn.relu(mask0_out)
        mask0_out = tf.expand_dims(mask0_out,1)
        mask0_out = tf.expand_dims(mask0_out,1)

        mask1_array = np.ones((1, class_num, 128, 1), np.float32)
        mask1_tensor = tf.get_variable('mask1', dtype=tf.float32, trainable=args.learned_mask,
                                      initializer=tf.constant(mask1_array))
        mask1_tensor = tf.tile(mask1_tensor, [batch_size, 1, 1, 1])
        control1_code = tf.tile(code_placeholder,[1,1,128,1])

        mask1_out = tf.multiply(mask1_tensor, control1_code)
        mask1_out = tf.reduce_sum(mask1_out, axis=1)
        mask1_out = tf.squeeze(mask1_out)
        mask1_out = tf.nn.relu(mask1_out)
        mask1_out = tf.expand_dims(mask1_out,1)
        mask1_out = tf.expand_dims(mask1_out,1)


        # Build the inference graph
        prelogits, _ = network.inference(image_batch, mask0_out, mask1_out, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)

        embeddings_pre = tf.multiply(mask_out, prelogits)

        embeddings = tf.nn.l2_normalize(embeddings_pre, 1, 1e-10, name='embeddings')
        anchor_index = list(range(0,batch_size,3))
        positive_index = list(range(1,batch_size,3))
        negative_index = list(range(2,batch_size,3))

        a_indice = tf.constant(np.array(anchor_index))
        p_indice = tf.constant(np.array(positive_index))

        n_indice = tf.constant(np.array(negative_index))

        anchor = tf.gather(embeddings,a_indice)
        positive = tf.gather(embeddings,p_indice)
        negative = tf.gather(embeddings,n_indice)

        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables())
        
        # Create a saver
        trainable_variables = tf.global_variables()
        saver = tf.train.Saver(trainable_variables, max_to_keep=35)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            Accuracy = [0]
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                # epoch = step // args.epoch_size
                # Train for one epoch
                code_list = []
                triplets_list = []
                max_num = 32768
                if (epoch+1)%args.lr_epoch == 0:
                    args.learning_rate = 0.1*args.learning_rate
                if args.random_trip:

                 for i in range(class_num):

                    code = np.zeros((batch_size, class_num, 1, 1), np.float32)
                    _class = i
                    code[:, _class, :, :] = 1

                    Triplets = triplet_random(args, sess, all_image_list[i], all_image_list, epoch, image_placeholder,
                                              batch_size_placeholder, learning_rate_placeholder,
                                              phase_train_placeholder, global_step,
                                              embeddings, total_loss, train_op, summary_op, summary_writer,
                                              args.learning_rate_schedule_file,
                                              args.embedding_size, anchor, positive, negative, triplet_loss, max_num)
                    triplets_list.append(Triplets)
                    code_list.append(code)

                else:
                  for i in range(class_num):

                      code = np.zeros((batch_size, 1, 1, 1), np.float32)
                      _class = i
                      if _class > 3:
                          _class = 3

                      code[:, :, :, :] = _class
                      print(class_name[i])
                      Triplets = triplet(args, sess, all_image_list[i], epoch, image_placeholder, code_placeholder,
                          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
                          embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                          args.embedding_size, anchor, positive, negative, triplet_loss,code)

                      triplets_num = len(Triplets)

                      triplets_list.append(Triplets)
                      code_list.append(code)



                train(args, sess, image_list, epoch, image_placeholder, code_placeholder,
                    batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
                    embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                    args.embedding_size, anchor, positive, negative, triplet_loss, triplets_list, code_list, model_dir, Accuracy)

                if (epoch+1)%2 == 0:
                    Accuracy = test(args, sess, image_list, epoch, image_placeholder,code_placeholder,
                          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
                          embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                          args.embedding_size, anchor, positive, negative, triplet_loss, triplets_list, Accuracy)

                # Save variables and the metagraph if it doesn't exist already
                model_name = 'epoch' + str(epoch+1)
                print(model_dir)
                if (epoch+1) > 0 :
                    if (epoch +1)%2 == 0:
                        save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step)
                        print('models are saved in ', os.path.join(model_dir, model_name))
                epoch = epoch + 1
    sess.close()
    return model_dir

def normalized_image(img_sym, resnet_mean = [102.9801, 115.9465, 122.7717]):
    return tf.scalar_mul(0.0078125, tf.subtract(tf.cast(img_sym, tf.float32), tf.constant(resnet_mean)))

def triplet_random(args, sess, dataset, image_list, epoch, image_placeholder,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss,max_num):

    batch_number = 0

    images_data = h5py.File('../face_hdf5_150by150/train_classification.h5')
    images = images_data['data']

    image_list_p = dataset[0]
    image_list_n = dataset[1]


    random.shuffle(image_list_n)

    start_time = time.time()
    nrof_examples = len(image_list_p)
    triplets = []
    nrof_neg = len(image_list_n)

    for j in range(nrof_examples):

        i = j
        a_idx = i
        p_idx = np.random.randint(i,nrof_examples)
        n_idx = np.random.randint(nrof_neg)
        triplets.append((image_list_p[a_idx], image_list_p[p_idx], image_list_n[n_idx]))



    return triplets

def triplet(args, sess, images_list, epoch, image_placeholder, code_placehoder,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss, code):

    batch_number = 0

    images_data = h5py.File('../face_hdf5_150by150/train_classification.h5')
    images = images_data['data']

    image_list_p = images_list[0]
    image_list_n = images_list[1]


    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    random.shuffle(image_list_n)

    start_time = time.time()
    nrof_examples = len(image_list_p)

    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    emb_p_array = np.zeros((0, embedding_size))
    emb_n_array = np.zeros((0, embedding_size))
    nrof_batches = int(np.ceil(nrof_examples / args.batch_size))

    image_mean = cv2.imread('../image_mean1.jpg')
    image_mean = cv2.resize(image_mean, (120,120))
    image_mean = image_mean.astype(np.float32)
    print('load mean image done!')
    print('start to select positive')
    for i in range(nrof_batches):
        batch_number = i
        print(i)
        # batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)

        batch_size = args.batch_size

        [image_p_array, image_list_p]= facenet.get_image_batch(images, image_list_p, batch_size, batch_number, image_mean, args)
        code_batch = code[0:batch_size]
        emb_p= sess.run([embeddings], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: lr, phase_train_placeholder: True, image_placeholder: image_p_array, code_placehoder: code_batch})
        emb_p_array = np.vstack((emb_p_array,emb_p[0]))
    print('number of positive batch:', i+1)

    nrof_n_examples = min(nrof_examples, 30000)

    nrof_n_batches = int(np.ceil(nrof_n_examples / args.batch_size))
    print('start to select negative')
    for j in range(nrof_n_batches):
        batch_number = j
        # batch_size = min(nrof_examples-j*args.batch_size, args.batch_size)
        print(j)
        batch_size = args.batch_size

        [image_n_array, image_list_n] = facenet.get_image_batch(images, image_list_n, batch_size, batch_number, image_mean, args)
        code_batch = code[0:batch_size]

        emb_n = sess.run([embeddings], feed_dict={batch_size_placeholder: batch_size,
                                                 learning_rate_placeholder: lr, phase_train_placeholder: True,
                                                 image_placeholder: image_n_array, code_placehoder: code_batch})
        emb_n_array = np.vstack((emb_n_array,emb_n[0]))
    print('number of negative batch:', j+1)


    print('compute feature is done!')
    print('%.3f' % (time.time()-start_time))
    # Select triplets based on the embeddings
    print('Selecting suitable triplets for training')
    triplets, nrof_random_negs, nrof_triplets = select_binary_triplets(emb_p_array, emb_n_array, image_list_p, image_list_n, args.alpha)
    selection_time = time.time() - start_time
    print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
        (nrof_random_negs, nrof_triplets, selection_time))
    return triplets


def train(args, sess, dataset, epoch, image_placeholder, code_placeholder,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss, triplets_list, code_list,model_dir,Accuracy):

    batch_number = 0

    images_data = h5py.File('../face_hdf5_150by150/train_classification.h5')
    images = images_data['data']

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)


    start_time = time.time()

    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    nrof_examples = 0
    class_num = len(triplets_list)
    # for i in range(class_num):
    nrof_examples = max(len(triplets_list[0]), nrof_examples)

    nrof_batches = int(np.ceil(nrof_examples / (args.batch_size/3/class_num)))


    print('load mean image done!')

    train_time = 0
    batch_number = 0
    batch_size = args.batch_size
        # emb_array = np.zeros((nrof_examples, embedding_size))
        # loss_array = np.zeros((nrof_triplets,))
    while batch_number < nrof_batches:

            start_time = time.time()

            class_num = len(triplets_list)

            image_array = np.zeros((batch_size,150,150,3),np.float32)
            code_array = np.zeros((0,class_num,1,1), np.float32)
            length = []
            for t in range(class_num):

              triplets = triplets_list[t]
              length.append(len(triplets))

              start_index = t*int(batch_size/class_num)

              image_array = facenet.get_triplet_image_batch1(images, triplets, batch_size/class_num, batch_number, image_array, start_index, args)
              code = code_list[t]

              code_array_tem = code[0:int(batch_size/class_num)]

              code_array = np.vstack((code_array, code_array_tem))

            time1 = time.time() - start_time
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True, image_placeholder: image_array, code_placeholder: code_array}
            err, _, step, emb = sess.run([loss, train_op, global_step, embeddings], feed_dict=feed_dict)
            # emb_array[lab,:] = emb
            # loss_array[i] = err
            emb_feature = emb
            nrof_pair = emb_feature.shape[0]/3
            correct_num = 0

            for pair_index in range(0, int(nrof_pair/class_num)):
                anchor_index = pair_index*3
                pos_index = anchor_index + 1
                neg_index = anchor_index + 2
                an_feature = emb_feature[anchor_index,:]
                pos_feature = emb_feature[pos_index, :]
                neg_feature = emb_feature[neg_index, :]
                if np.sum(np.square(an_feature - pos_feature),0) < np.sum(np.square(an_feature - neg_feature),0):
                    correct_num = correct_num + 1

            p_dist = np.sum(np.square(an_feature - pos_feature), 0)
            n_dist = np.sum(np.square(an_feature - neg_feature),0)
            print('triplet number:', length)
            print('pos_distance and neg_distance', p_dist, n_dist, 'model_dir:',model_dir,'last accuracy:',Accuracy[-1])
            accuracy = correct_num/(nrof_pair/class_num)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tread_time %.3f\tAccuracy %2.3f\tlr %2.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, nrof_batches, duration, time1, accuracy, lr, err))
            batch_number += 1
            train_time += duration

        # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    #summary.value.add(tag='time/selection', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step
def get_code_batch(code, triplets, batch_size, batch_index):

    nrof_examples = len(triplets)

    j = batch_index * batch_size % nrof_examples

    if j + batch_size < nrof_examples:
        code_array = code[j: j+ batch_size]

    else:
        code_array1 = code[j:nrof_examples]
        code_array2 = code[0:batch_size - (nrof_examples-j)]
        code_array = np.vstack((code_array1,code_array2))
    return code_array
def test(args, sess, dataset, epoch, image_placeholder, code_placeholder,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss, triplet_list, Accuracy):


    images_data = h5py.File('../face_hdf5_150by150/test_triplet_smile.h5')
    images_a = images_data['anchors']
    images_p = images_data['positive']
    images_n = images_data['negative']

    nrof_images = images_a.shape[0]
    nrof_batches = int(math.ceil(1.0 * nrof_images / (args.batch_size/3)))

    batch_number = 0
    correct_num = 0
    class_num = len(triplet_list)
    batch_size = args.batch_size

   # image_mean = cv2.imread('../image_mean1.jpg')
   # image_mean = cv2.resize(image_mean, (120, 120))
   # image_mean = image_mean.astype(np.float32)
    print('load mean image done!')
    # fail_dir = '../fail_smile_pair'
    code = np.zeros((batch_size, class_num, 1, 1), np.float32)
    _class = 0
    if _class > 3:
        _class = 3

    code[:, _class, :, :] = 1

    check = 0
    while batch_number < nrof_batches:

            # batch_size = min(nrof_images - batch_number * args.batch_size, args.batch_size)
            batch_inter = int(batch_size/3)
            if (nrof_images - batch_number*batch_inter) < batch_inter:
                check = check + 1
                anchor_array1 = images_a[batch_number * batch_inter:nrof_images]
                anchor_array2 = images_a[0:batch_inter - (nrof_images - batch_number * batch_inter)]
                anchor_array = np.vstack((anchor_array1,anchor_array2))

                positive_array1 = images_p[batch_number * batch_inter:nrof_images]
                positive_array2 = images_p[0:batch_inter - (nrof_images - batch_number * batch_inter)]
                positive_array = np.vstack((positive_array1, positive_array2))

                negative_array1 = images_n[batch_number * batch_inter:nrof_images]
                negative_array2 = images_n[0:batch_inter - (nrof_images - batch_number * batch_inter)]
                negative_array = np.vstack((negative_array1, negative_array2))
            else:
                anchor_array = images_a[batch_number*batch_inter:batch_number*batch_inter + batch_inter]
                positive_array = images_p[batch_number * batch_inter:batch_number * batch_inter + batch_inter]
                negative_array = images_n[batch_number * batch_inter:batch_number * batch_inter + batch_inter]

            image_array = np.vstack((anchor_array,positive_array,negative_array))


            feed_dict = {batch_size_placeholder: batch_size, phase_train_placeholder: False, image_placeholder: image_array, code_placeholder: code}

            emb = sess.run(embeddings, feed_dict=feed_dict)

            batch_number += 1

            # emb_array[lab,:] = emb
            # loss_array[i] = er

            for i in range(int(batch_size/3)):

                an_feature = emb[i,:]
                pos_feature = emb[i+int(batch_size/3), :]
                neg_feature = emb[i+2*int(batch_size/3), :]

                if np.sum(np.square(an_feature - pos_feature),0) < np.sum(np.square(an_feature - neg_feature),0):
                    correct_num = correct_num + 1
                # else:
                #     fail_anchor_im = anchor_array[i,:,:,:] + 70
                #     fail_pos_im = positive_array[i,:,:,:] + 70
                #     fail_neg_im = negative_array[i,:,:,:] + 70
                #     anchor_name = str(batch_number)+ '_' + str(i) + 'anchor.jpg'
                #
                #     pos_name = str(batch_number) + '_' + str(i) + 'pos.jpg'
                #
                #     neg_name = str(batch_number) + '_' + str(i) + 'neg.jpg'

                    #
                    # cv2.imwrite(os.path.join(fail_dir,anchor_name), fail_anchor_im)
                    # cv2.imwrite(os.path.join(fail_dir,pos_name), fail_pos_im)
                    # cv2.imwrite(os.path.join(fail_dir,neg_name), fail_neg_im)

            # print(batch_number, nrof_batches)
            # p_dist = np.sum(np.square(an_feature - pos_feature), 0)
            # n_dist = np.sum(np.square(an_feature - neg_feature),0)
            # print('pos_distance and neg_distance', p_dist, n_dist)
    nrof_images = nrof_batches*batch_inter
    accuracy = correct_num/nrof_images
    Accuracy.append(accuracy)
    print('accuracy:', Accuracy)
    return Accuracy


def select_binary_triplets(embeddings_p, embeddings_n,  image_p_paths, image_n_paths, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    nrof_images = embeddings_p.shape[0]
    print('start to select triplets')
    for j in range(nrof_images):
        a_idx = j

        neg_dists_sqr = np.sum(np.square(embeddings_p[a_idx] - embeddings_n), 1)

        # for pair in xrange(j, nrof_images):  # For every possible positive pair.
        pair = np.random.randint(j, nrof_images)
        p_idx = emb_start_idx + pair
        pos_dist_sqr = np.sum(np.square(embeddings_p[a_idx,:] - embeddings_p[p_idx,:]))

        all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
        hard_neg = np.where(neg_dists_sqr - pos_dist_sqr < 0)[0]
        nrof_random_negs = all_neg.shape[0]
        nrof_hard_negs = hard_neg.shape[0]
        if nrof_random_negs > 0:

            if nrof_hard_negs > 0:
                rnd_idx = np.random.randint(nrof_hard_negs)
                n_idx = hard_neg[rnd_idx]
            else:
                rnd_idx = np.random.randint(nrof_random_negs)
                n_idx = all_neg[rnd_idx]

            triplets.append((image_p_paths[a_idx], image_p_paths[p_idx], image_n_paths[n_idx]))
            # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
            #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
            trip_idx += 1

        num_trips += 1

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def select_binary_triplets_gpu(embeddings_p, embeddings_n,  image_p_paths, image_n_paths, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    nrof_images = embeddings_p.shape[0]
    print('start to select triplets')
    for j in range(nrof_images):
        a_idx = j

        neg_dists_sqr = np.sum(np.square(embeddings_p[a_idx] - embeddings_n), 1)

        # for pair in xrange(j, nrof_images):  # For every possible positive pair.
        pair = np.random.randint(j, nrof_images)
        p_idx = emb_start_idx + pair
        pos_dist_sqr = np.sum(np.square(embeddings_p[a_idx,:] - embeddings_p[p_idx,:]))

        all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
        hard_neg = np.where(neg_dists_sqr - pos_dist_sqr < 0)[0]
        nrof_random_negs = all_neg.shape[0]
        nrof_hard_negs = hard_neg.shape[0]
        if nrof_random_negs > 0:

            if nrof_hard_negs > 0:
                rnd_idx = np.random.randint(nrof_hard_negs)
                n_idx = hard_neg[rnd_idx]
            else:
                rnd_idx = np.random.randint(nrof_random_negs)
                n_idx = all_neg[rnd_idx]

            triplets.append((image_p_paths[a_idx], image_p_paths[p_idx], image_n_paths[n_idx]))
            # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
            #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
            trip_idx += 1

        num_trips += 1

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool,default= True)
    parser.add_argument('--lr_epoch', type=int,default= 30)
    parser.add_argument('--random_trip', type=bool, default= False)
    parser.add_argument('--experiment_name', type=str,default='triplet')
    parser.add_argument('--category', type=int, default=31)
    parser.add_argument('--learned_mask', type=bool, default=True)
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', default = '')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=40)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--num_attribute', type=int,
        help='', default=1)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
