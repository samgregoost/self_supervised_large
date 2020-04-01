from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import TensorflowUtils as utils
import read_MITSceneParsingDataParis as scene_parsing
import datetime
import BatchDatsetReaderCfar as dataset
from six.moves import xrange
import math
from scipy import signal
from scipy.interpolate import interp1d

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "20", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "/scratch1/ram095/nips20/logs_stl40randedgew/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/scratch1/ram095/nips20/datasets/stl", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 3
IMAGE_SIZE = 64


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

'''
def decoder(image):
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("decoder"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)
        
    return pool5


'''
    

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    try:
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    except ValueError as err:
        msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
        err.args = err.args + (msg,)
        raise
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def new_conv_layer( bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu



def new_deconv_layer(bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

def batchnorm(bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):

            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)


            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed


def inference(images, keep_prob,z,e,is_train):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """

    encoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    encoderLayerNum = encoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1 
    print("encoderLayerNum=", encoderLayerNum)
    encoderLayerNum = encoderLayerNum

    decoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    decoderLayerNum = decoderLayerNum - 1
    print("decoderLayerNum=", decoderLayerNum)
    decoderLayerNum = decoderLayerNum
    print("setting up vgg initialized conv layers ...")
    #model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    #mean = model_data['normalization'][0][0][0]
    #mean_pixel = np.mean(mean, axis=(0, 1))

    #weights = np.squeeze(model_data['layers'])

    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):

        previousFeatureMap = images
        previousDepth = 3
        depth = 64

        for layer in range(1, encoderLayerNum):
            print("build_reconstruction encoder layer=", layer)
            conv = tf.nn.dropout(new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv" + str(layer))),keep_prob)
            bn = tf.nn.leaky_relu(batchnorm(conv, is_train, name=("bn" + str(layer))))
            previousFeatureMap = bn
            previousDepth = depth
            depth = depth * 2

            # last layer
        conv = new_conv_layer(previousFeatureMap, [4,4,previousDepth,4000], stride=2, padding='VALID', name=('conv' + str(encoderLayerNum)))
        bn = tf.nn.leaky_relu(batchnorm(conv, is_train, name=("bn" + str(encoderLayerNum))))

        previousDepth = 4000
        depth = 64 * pow(2,decoderLayerNum-2)
        featureMapSize = 4

        deconv =  tf.nn.dropout(new_deconv_layer( bn, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], padding='VALID', stride=2, name=("deconv" + str(decoderLayerNum))),keep_prob)

        #debn_ = tf.nn.relu(batchnorm(deconv, is_train, name=("debn" + str(decoderLayerNum)))) 
        z_ = z/tf.norm(z)
        debn_ = tf.nn.relu(batchnorm(deconv, is_train, name=("debn" + str(decoderLayerNum))))
        debn = tf.concat([debn_,tf.tile(z_,[1,4,4,1])],axis = 3) + e

    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
        print("#################################")
        print(debn)
        previousFeatureMap = debn
        previousDepth = 552
        depth = depth / 2
        featureMapSize = featureMapSize *2

        for layer in range(decoderLayerNum-1,1, -1):
            print("build_reconstruction decoder layer=", layer)
            deconv = new_deconv_layer( previousFeatureMap, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv" + str(layer)))
            debn = tf.nn.relu(batchnorm(deconv, is_train, name=('debn'+ str(layer))))
            previousFeatureMap = debn
            previousDepth = depth
            depth = depth / 2
            featureMapSize = featureMapSize *2

        recon = tf.nn.tanh(new_deconv_layer( debn, [4,4,3,previousDepth], [FLAGS.batch_size,64,64,3], stride=2, name="recon"))


        ''' 
        conv1 = new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
        bn1 = tf.nn.leaky_relu(batchnorm(conv1, is_train, name='bn1'))
        conv2 = new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2" )
        bn2 = tf.nn.leaky_relu(batchnorm(conv2, is_train, name='bn2'))
        conv3 = new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3")
        bn3 = tf.nn.leaky_relu(batchnorm(conv3, is_train, name='bn3'))
        conv4 = new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4")
        bn4 = tf.nn.leaky_relu(batchnorm(conv4, is_train, name='bn4'))
        conv5 = new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5")
        bn5 = tf.nn.leaky_relu(batchnorm(conv5, is_train, name='bn5'))
        conv6 = new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6')
        bn6 = tf.nn.leaky_relu(batchnorm(conv6, is_train, name='bn6'))
    

        deconv4 = new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4")
        debn4 = tf.nn.relu(batchnorm(deconv4, is_train, name='debn4'))
        deconv3 = new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
        debn3 = tf.nn.relu(batchnorm(deconv3, is_train, name='debn3'))
        deconv2 = new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2")
        debn2 = tf.nn.relu(batchnorm(deconv2, is_train, name='debn2'))
        deconv1 = new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1")
        debn1 = tf.nn.relu(batchnorm(deconv1, is_train, name='debn1'))
        recon = new_deconv_layer( debn1, [4,4,3,64], [batch_size,64,64,3], stride=2, name="recon")

        print("##########################################")
        print(recon)
        '''
    return recon, debn_

def predictor_(h,z, is_train):
    z_tiled = tf.tile(z,[1,4,4,1])
    concat = tf.concat([h,z_tiled],axis = 3)
    conv1 = new_conv_layer(concat, [3,3,1024,512], stride=1, padding='VALID', name=('pred_conv_1'))
    bn = tf.nn.leaky_relu(batchnorm(conv1, is_train, name=("pred_bn_1")))
    bn_ln = tf.reshape(bn,[FLAGS.batch_size,-1])
    fc1 = tf.expand_dims(tf.expand_dims(tf.layers.dense(bn_ln,10),1),1)
   # bn2 = tf.nn.leaky_relu(batchnorm(fc1, is_train, name=("pred_bn_2")))
   # z_pred = tf.clip_by_value(tf.nn.tanh(fc1),-0.1,0.1)
    z_pred = fc1
    return z_pred

def predictor(h,z,e, is_train):
 #   z_tiled = tf.tile(z,[1,4,4,1])
    with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE):
        concat = tf.concat([tf.contrib.layers.flatten(h),tf.contrib.layers.flatten(z)],axis = 1, name = "pred_concat") + tf.reshape(e,[FLAGS.batch_size,-1])
#    concat = tf.reshape(tf.concat([h,z_tiled],axis = 3),[FLAGS.batch_size,-1])
        fc1 = tf.nn.leaky_relu(tf.layers.dense(concat,512), name = "pred_fc1")
 #   bn = tf.nn.leaky_relu(batchnorm(fc1, is_train, name=("pred_bn_1")))
   
        fc2 = tf.nn.leaky_relu(tf.layers.dense(fc1,512), name = "pred_fc2")
  #  bn2 = tf.nn.leaky_relu(batchnorm(fc2, is_train, name=("pred_bn_2")))

        fc3 = tf.expand_dims(tf.expand_dims(tf.layers.dense(fc2,40),1),1, name = "pred_fc3")
   # bn2 = tf.nn.leaky_relu(batchnorm(fc1, is_train, name=("pred_bn_2")))
   # z_pred = tf.clip_by_value(tf.nn.tanh(fc1),-0.1,0.1)
        z_pred = tf.nn.tanh(fc3)
    return z_pred


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def train_predictor(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def train_z(loss,Z):
    return tf.gradients(ys = loss, xs = Z)

def random_mask(input_size):
    x1 =  random.randint(5, 20)
    w1 =  random.randint(20, 34)
    y1 =  random.randint(5, 20)
    h1 =  random.randint(20, 34)

    mask = np.zeros((1,64,64,1))
    mask[:,x1:x1+w1,y1:y1+h1,:] = 1.0

    mask2 = np.zeros((1,64,64,1))
    mask2[:,x1-5:x1+w1+5,y1-5:h1+y1+5,:] = 1.0
    mask2 = mask2 - mask

    return mask, mask2

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="annotation")
    z = tf.placeholder(tf.float32, shape=[None, 1, 1, 40], name="z")
    mask = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="mask")
    mask2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="mask2")
    z_new = tf.placeholder(tf.float32, shape=[None, 1, 1, 40], name="z_new")
    istrain = tf.placeholder(tf.bool)
   #z_lip = tf.placeholder(tf.float32, shape=[None, 1, 1, 10], name="z_lip")
   #z_lip_inv = tf.placeholder(tf.float32, shape=[None, 1, 1, 10], name="z_lip_inv")
    e = tf.placeholder(tf.float32, shape=[None, 4, 4, 552], name="e")
    e_p = tf.placeholder(tf.float32, shape=[None, 1, 1, 8232], name="e_p")
    

    # pred_annotation, logits = inference(image, keep_probability,z)
 #   tf.summary.image("input_image", image, max_outputs=2)
 #   tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
 #   tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
#    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
 #                                                                         labels=tf.squeeze(annotation, squeeze_dims=[3]),
  #                                                                    name="entropy")))
    
    
   # mask_ = tf.ones([FLAGS.batch_size,32,64,3])
   # mask = tf.pad(mask_, [[0,0],[0,32],[0,0],[0,0]])

 #   mask2__ = tf.ones([FLAGS.batch_size,78,78,3])
  #  mask2_ = tf.pad(mask2__, [[0,0],[25,25],[25,25],[0,0]])
   # mask2 = mask2_ - mask
    zero = tf.zeros([20,1,1,8232])   
    logits, h  = inference((1-mask)*image + mask*1.0, keep_probability,z,0.0,istrain)
    logits_e, h_e = inference((1-mask)*image + mask*1.0, keep_probability,z,e,istrain)
    #logits_lip,_  = inference((1-mask)*image + mask*0.0, keep_probability,z_lip,istrain   ) 
    #logits_lip_inv,_  = inference((1-mask)*image + mask*0.0, keep_probability,z_lip_inv,istrain   )

    z_pred = predictor(h,z,zero,istrain)
    z_pred_e = predictor(h,z,e_p,istrain)
  #  z_pred_lip =  predictor(h,z_lip,istrain)
  #  z_pred_lip_inv =  predictor(h,z_lip_inv,istrain)
   # logits = inference(image, keep_probability,z,istrain)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
   # tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    
   # lossz = 0.1 * tf.reduce_mean(tf.reduce_sum(tf.abs(z),[1,2,3]))
  #  lossz = 0.1 * tf.reduce_mean(tf.abs(z))
   # loss_all = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((image - logits)),[1,2,3])))
   # loss_all = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(image - logits)),1))
    
  #  loss_mask = 0.8*tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((image - logits)*mask),[1,2,3])))
    loss_mask = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((annotation - logits)*mask)),1))
    loss_mask2 = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((annotation - logits)*mask2)),1)) 
    loss_ =  loss_mask + loss_mask2
  #  loss = tf.reduce_mean(tf.squared_difference(logits ,annotation ))
    loss_summary = tf.summary.scalar("entropy", loss_)
   # zloss = tf.reduce_mean(tf.losses.cosine_distance(tf.contrib.layers.flatten(z_new) ,tf.contrib.layers.flatten(z_pred),axis =1)) 
    zloss_ = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((z_pred - z_new))),1))
 #   zloss_lip = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((z_pred - z_pred_lip))),1))
#    zloss_lip_inv = -tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((z_pred - z_pred_lip_inv))),1))
 
#    z_loss = zloss_ + 0.1* zloss_lip# + zloss_lip_inv
        

    lip_loss_dec = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((logits - logits_e))),1))
    loss = loss_ + 0.1*lip_loss_dec 
   
    lip_loss_pred = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((z_pred - z_pred_e))),1))     
    zloss = zloss_ + 0.1*lip_loss_pred
   
    grads = train_z(loss_mask,z)    

    trainable_var = tf.trainable_variables()
    trainable_z_pred_var = tf.trainable_variables(scope="predictor")
    trainable_d_pred_var = tf.trainable_variables(scope="decoder")


    print(trainable_z_pred_var)
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)
    train_pred = train_predictor(zloss,trainable_z_pred_var)
    

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    
    saved =True
    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            print(np.max(train_images))
          #  z_ = np.reshape(signal.gaussian(200, std=1),(FLAGS.batch_size,1,1,10))-0.5
            z_ = np.random.uniform(low=-1.0, high=1.0, size=(FLAGS.batch_size,1,1,40))
 #           train_images[train_images < 0.] = -1.
  #          train_annotations[train_annotations < 0.] = -1.
   #         train_images[train_images >= 0.] = 1.0
    #        train_annotations[train_annotations >= 0.] = 1.0
                    
            x1 = random.randint(0, 10) 
            w1 = random.randint(30, 54)
            y1 = random.randint(0, 10)
            h1 = random.randint(30, 54)

            cond = random.randint(0, 10)
           # saved = True   
            if False:
                saved = False
                train_images_m, train_annotations_m = train_dataset_reader.get_random_batch(FLAGS.batch_size)
                train_images_m[train_images_m < 0.] = -1.
                train_annotations_m[train_annotations_m < 0.] = -1.
                train_images_m[train_images_m >= 0.] = 1.0
                train_annotations_m[train_annotations_m >= 0.] = 1.0

                train_images = (train_images + 1.)/2.0*255.0
                train_annotations = (train_annotations + 1.)/2.0*255.0
                train_images_m = (train_images_m + 1.)/2.0*255.0
                train_annotations_m = (train_annotations_m +  1.)/2.0*255.0

                train_images_m[:,32:,:,:] = 0
                train_annotations_m[:,32:,:,:] = 0
                train_images = np.clip((train_images + train_images_m),0.0,255.0)
                train_annotations =  np.clip((train_annotations + train_annotations_m),0.0,255.0)
                '''
                train_images[train_images < 0.] = -1.
                train_annotations[train_annotations < 0.] = -1.
                train_images[train_images >= 0.] = 1.0
                train_annotations[train_annotations >= 0.] = 1.0
                '''

                train_annotations_ = np.squeeze(train_annotations,axis = 3)
                train_images_ = train_images

                train_images = train_images/127.5 - 1.0
                train_annotations = train_annotations/127.5 - 1.0                     

               # for itr_ in range(FLAGS.batch_size):
                #    utils.save_image(train_images_[itr_].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr_) )
                 #   utils.save_image(train_annotations_[itr_].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr_) )
    #        train_images[:,x1:w1,y1:h1,:] = 0
            
          #  print(train_images)
            r_m, r_m2 = random_mask(64)
           #feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, z: z_,mask:r_m, istrain:True }
           #train_images[:,50:100,50:100,:] =0
            v = 0
           # print(train_images)    
            error_dec =  np.random.normal(0.0,0.001,(FLAGS.batch_size,4,4,552))
            error_dec_ =  np.random.normal(0.0,0.001,(FLAGS.batch_size,1,1,8232))
           # z_l_inv = z_ + np.random.normal(0.0,0.1)
          #  feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, z: z_, e:error_dec, mask:r_m, istrain:True }
          #  z_l = z_ + np.random.normal(0.0,0.001)
      #     lloss,_ = sess.run([lip_loss, train_lip ], feed_dict=feed_dict)
           # z_l = z_ + np.random.normal(0.0,0.001)
           # print("Step: %d, lip_loss:%g" % (itr,lloss))

            for p in range(20):
                z_ol = np.copy(z_)
            #    z_l = z_ol + np.random.normal(0.0,0.001)
               # print("666666666666666666666666666666666666666")
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, z: z_,e:error_dec, mask:r_m,mask2:r_m2, istrain:True }         
               # lloss,_ = sess.run([lip_loss, train_lip ], feed_dict=feed_dict)
               # print("Step: %d, z_step: %d, lip_loss:%g" % (itr,p,lloss))     
                z_loss, summ = sess.run([loss,loss_summary], feed_dict=feed_dict)
                print("Step: %d, z_step: %d, Train_loss:%g" % (itr,p,z_loss))
#                print(z_) 
                g = sess.run([grads],feed_dict=feed_dict)
                v_prev = np.copy(v)
               # print(g[0][0].shape)
                v = 0.001*v - 0.1*g[0][0]
                z_ += 0.001 * v_prev + (1+0.001)*v
                z_ = np.clip(z_, -10.0, 10.0)
                
                '''
                m = interp1d([-10.0,10.0],[-1.0,1.0])
                print(np.max(z_))
                print(np.min(z_))
                z_ol_interp = m(z_ol)
                z_interp = m(z_)
                _,z_pred_loss =sess.run([train_pred,zloss],feed_dict={image: train_images,mask:r_m,z:z_ol_interp,z_new:z_interp,e_p:error_dec_,istrain:True,keep_probability: 0.85})
                print("Step: %d, z_step: %d, z_pred_loss:%g" % (itr,p,z_pred_loss))
                '''

               # _,z_pred_loss =sess.run([train_pred,zloss],feed_dict={image: train_images,mask:r_m,z:z_ol,z_new:z_,istrain:True,keep_probability: 0.85})
               # print("Step: %d, z_step: %d, z_pred_loss:%g" % (itr,p,z_pred_loss))
               # z_ = np.clip(z_, -1.0, 1.0)
               # print(v.shape)
               # print(z_.shape)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability:0.85,mask:r_m,e:error_dec, z: z_,mask2:r_m2, istrain:True }
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                
                train_writer.add_summary(summary_str, itr)
              

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
     #           valid_annotations[valid_annotations < 0.] = -1.
      #          valid_images[valid_images < 0.] = -1.
       #         valid_annotations[valid_annotations >= 0.] = 1.0
        #        valid_images[valid_images >= 0.] = 1.0
                
                x1 = random.randint(0, 10)
                w1 = random.randint(30, 54)
                y1 = random.randint(0, 10)
                h1 = random.randint(30, 54)

     #           valid_images[:,x1:w1,y1:h1,:] = 0
                 
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images,mask:r_m, annotation: valid_annotations,
                                                       keep_probability: 1.0, z: z_,e:error_dec, istrain:False,mask2:r_m2 })
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model_z_center_7.ckpt", 500)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(20)
      #  valid_annotations[valid_annotations < 0.] = -1.0
      #  valid_images[valid_images < 0.] = -1.0
      #  valid_annotations[valid_annotations >= 0.] = 1.0
      #  valid_images[valid_images >= 0.] = 1.0
        
        x1 = random.randint(0, 10)
        w1 = random.randint(30, 54)
        y1 = random.randint(0, 10)
        h1 = random.randint(30, 54)

      #  valid_images[:,x1:w1,y1:h1,:] = 0
        r_m, r_m2 = random_mask(64)      
       # z_ = np.zeros(low=-1.0, high=1.0, size=(FLAGS.batch_size,1,1,10))
       # z_ = np.reshape(signal.gaussian(200, std=1),(FLAGS.batch_size,1,1,10))-0.5
        z_ = np.random.uniform(low=-1.0, high=1.0, size=(FLAGS.batch_size,1,1,40))
        feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 0.85, z: z_, istrain:False,mask:r_m,mask2:r_m2 }
        v= 0
        m__ = interp1d([-10.0,10.0],[-1.0,1.0])
        z_ = m__(z_)
  #      feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 0.85, z: z_, istrain:False,mask:r_m }
        for p in range(20):
                z_ol = np.copy(z_)
               # print("666666666666666666666666666666666666666")
#                print(z_)
               # feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 0.85, z: z_, istrain:False,mask:r_m }
              #  z_loss, summ = sess.run([loss,loss_summary], feed_dict=feed_dict)
              #  print("z_step: %d, Train_loss:%g" % (p,z_loss))
               # z_, z_pred_loss = sess.run(z_pred,zlossfeed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0, z:z_ol, istrain:False,mask:r_m})
                
#                print(z_)
                g = sess.run([grads],feed_dict=feed_dict)
                v_prev = np.copy(v)
               # print(g[0][0].shape)
                v = 0.001*v - 0.1*g[0][0]
                z_ = z_ol +  0.001 * v_prev + (1+0.001)*v
              #  z_ = z_ol + 0.001 * v_prev + (1+0.001)*v
               # print("z_____________")
               # print(z__)
               # print("z_")
               # print(z_)
            #    m__ = interp1d([-10.0,10.0],[-1.0,1.0])
           #     z_ol = m__(z_ol)
         #       z_ = sess.run(z_pred,feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 0.85, z:z_ol, istrain:False,mask:r_m})
             #   m_ = interp1d([-1.0,1.0],[-10.0,10.0])
              #  z_ = m_(z_)             
               # z_ = np.clip(z_, -1.0, 1.0)
               # print(z_pred_loss)
       # m_ = interp1d([-1.0,1.0],[-10.0,10.0])
       # z_ = m_(z_)
       
        pred = sess.run(logits, feed_dict={image: valid_images, annotation: valid_annotations,z:z_, istrain:False,mask:r_m,mask2:r_m2,
                                                    keep_probability: 0.85})
        

                
        valid_images_masked = ((1-r_m)*valid_images + 1.)/2.0*255
       # valid_images = (valid_images +1.)/2.0*255
       # predicted_patch = sess.run(mask) * pred
       # pred = valid_images_masked + predicted_patch 
        pred_ = (pred +1.)/2.0*255
#        pred = pred + 1./2.0*255
        
        pred = valid_images_masked *(1-r_m) + pred_ * r_m
        valid_annotations_ = (valid_annotations +1.)/2.0*255
       # pred = np.squeeze(pred, axis=3)
        print(np.max(pred))
        print(valid_images.shape)
        print(valid_annotations.shape)
        print(pred.shape)
       # for itr in range(FLAGS.batch_size):
           # utils.save_image(valid_images_masked[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
           # utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
           # utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="predz_" + str(5+itr))
            #        utils.save_image(valid_images_masked[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr)+'_' + str(p) )
             #       utils.save_image(valid_annotations_[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr)+'_' + str(p)  )
             #  utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="predz_" + str(5+itr)+'_' + str(p)  )
            #   print("Saved image: %d" % itr)
        
        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images_masked[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr) )
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="predz_" + str(5+itr) )     
            utils.save_image(valid_annotations_[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr) )        

if __name__ == "__main__":
    tf.app.run()
