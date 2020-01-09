import tensorflow as tf
import tf_slim as slim
import numpy as np
from multiprocessing import Process, Queue
import os
import numpy.random as nr
try:
    from skimage.transform import rotate, resize
except:
    print('warning: skimage not installed, disable rotation')
from scipy.io import loadmat
from time import time
import json



NET_ID = 0  # 0 for step3 pre-trained model, 1 for step2 pre-trained model


def spatial_softmax(fm):
    fm_shape = _get_tensor_shape(fm)
    n_grids = fm_shape[1] ** 2
    # transpose feature map
    fm = tf.transpose(a=fm, perm=[0, 3, 1, 2])
    t_fm_shape = _get_tensor_shape(fm)
    fm = tf.reshape(fm, shape=[-1, n_grids])
    # apply softmax
    prob = tf.nn.softmax(fm)
    # reshape back
    prob = tf.reshape(prob, shape=t_fm_shape)
    prob = tf.transpose(a=prob, perm=[0, 2, 3, 1])
    return prob


def _get_tensor_shape(x):
    s = x.get_shape().as_list()
    return [i if i is not None else -1 for i in s]


def attentionNet(inputs, pool_method='sigmoid'):
    assert(pool_method in ['sigmoid', 'softmax'])
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.compat.v1.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * 0.0005),
                        trainable=True):
        net = slim.conv2d(inputs, 256, [1, 1], padding='SAME', scope='conv1')
        if pool_method == 'sigmoid':
            net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='conv2')
        else:
            net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2')
            net = spatial_softmax(net)
    return net


def sketch_a_net_sbir(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.compat.v1.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * 0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')  # trainable=trainable
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            conv5 = slim.flatten(conv5)
            fc6 = slim.fully_connected(conv5, 512, trainable=trainable, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=trainable, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, axis=1)
    return fc7


def sketch_a_net_dssa(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.compat.v1.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * 0.0005),
                        trainable=False):  # when test 'trainable=True', don't forget to change it
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', trainable=trainable, scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            # residual attention
            att_mask = attentionNet(conv5, 'softmax')
            att_map = tf.multiply(conv5, att_mask)
            att_f = tf.add(conv5, att_map)
            attended_map = tf.reduce_sum(input_tensor=att_f, axis=[1, 2])
            attended_map = tf.nn.l2_normalize(attended_map, axis=1)
            att_f = slim.flatten(att_f)
            fc6 = slim.fully_connected(att_f, 512, trainable=trainable, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=trainable, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, axis=1)
            # coarse-fine fusion
            final_feature_map = tf.concat(1, [fc7, attended_map])
    return final_feature_map


def init_variables(model_file='./model/sketchnet_init.npy'):
    if NET_ID == 0:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch']
    else:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1']
    d = np.load(model_file, encoding="latin1").item()
    init_ops = []  # a list of operations
    for var in tf.compat.v1.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                print('Initialise var %s with weight %s' % (var.name, w_name))
                try:
                    if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        # init_ops.append(var.assign(d[w_name+'/weights:0']))
                        init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        # init_ops.append(var.assign(d[w_name+'/biases:0']))
                        init_ops.append(var.assign(d[w_name]['biases']))
                except KeyError:
                    if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name+'/weights:0']))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name+'/biases:0']))
                        # init_ops.append(var.assign(d[w_name]['biases']))
                except:
                    if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name][0]))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name][1]))
                        # init_ops.append(var.assign(d[w_name]['biases']))
    return init_ops


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.sub(x, y))
    d = tf.sqrt(tf.reduce_sum(input_tensor=d))  # What about the axis ???
    return d


def square_distance(x, y):
    return tf.reduce_sum(input_tensor=tf.square(x - y), axis=1)


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    with tf.compat.v1.name_scope("triplet_loss"):
        d_p_squared = square_distance(anchor_feature, positive_feature)
        d_n_squared = square_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        return tf.reduce_mean(input_tensor=loss), tf.reduce_mean(input_tensor=d_p_squared), tf.reduce_mean(input_tensor=d_n_squared)


class TripletSamplingLayer(object):
    def setup(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        """Setup the TripletSamplingLayer."""
        self.create_sample_fetcher(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)

    def create_sample_fetcher(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        self._blob_queue = Queue(10)
        self._prefetch_process = TripletSamplingDataFetcher(self._blob_queue, sketch_dir, image_dir, triplet_path, mean,
                                                            hard_ratio, batch_size, phase)
        self._prefetch_process.start()

        def cleanup():
            print('Terminating BlobFetcher')
            self._prefetch_process.terminate()
            self._prefetch_process.join()

        import atexit
        atexit.register(cleanup)

    def get_next_batch(self):
        return self._blob_queue.get()


class Transformer:
    def __init__(self, crop_size, num_channels, mean_=None, is_train=False, rotate_amp=None):
        self._crop_size = crop_size
        self._in_size = 256
        self._boarder_size = self._in_size - self._crop_size
        self._num_channels = num_channels
        self._is_train = is_train
        self._rotate_amp = rotate_amp
        if self._num_channels > 1 and self._rotate_amp > 0:
            raise Exception("can not rotate color image")
        if type(mean_) == str:
            mean_mat = np.load(mean_)
            self._mean = mean_mat.mean(axis=-1).mean(axis=-1).reshape(1, 3, 1, 1)  # mean value
        else:
            self._mean = mean_

    # @profile
    def transform(self, im):
        if len(im.shape) == 1:
            im = im.reshape((self._in_size, self._in_size)) if (self._num_channels == 1) else \
                im.reshape((self._num_channels, self._in_size, self._in_size))
        # rotation
        im1 = rand_rotate(im, self._rotate_amp) if self._rotate_amp is not None else im

        # translation and flip
        if len(im1.shape) == 2:  # gray scale
            im1 = im1.reshape((1, im1.shape[0], im1.shape[1]))
        x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
        y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2

        if nr.random() > 0.5 and self._is_train:
            im2 = im1[:, y:y+self._crop_size, x+self._crop_size:x:-1]
        else:
            im2 = im1[:, y:y+self._crop_size, x:x+self._crop_size]
        return im2

    def transform_all(self, imlist):
        processed = []
        for im in imlist:
            if im.shape[-1] == self._crop_size:
                processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                continue
            # translation and flip for image
            im = im.reshape(1, self._in_size, self._in_size, self._num_channels)
            x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            if nr.random() > 0.5 and self._is_train:
                trans_image = im[:, y:y+self._crop_size, x+self._crop_size:x:-1, :]
            else:
                trans_image = im[:, int(y):int(y+self._crop_size), int(x):int(x+self._crop_size), :]
            processed.append(trans_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
        # data = np.concatenate(processed, axis=0)
        data = np.reshape(processed, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data

    def transform_all_with_bbox(self, imlist, bbox, flag='anc'):
        # print 'transform'
        processed = []
        for id, im in enumerate(imlist):
            if im.shape[-1] == self._crop_size:
                processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels+1))
                continue

            def boundary_check(bbox):
                x, y, w, h = bbox
                x = min(max(0, x), 250)
                y = min(max(0, y), 250)
                w = min(max(5, w), 256 - x)
                h = min(max(5, h), 256 - y)
                return [x, y, w, h]

            def expand_boxes(bbox, ratio=1.2):
                x, y, w, h = bbox
                x_cen = x + 0.5 * w
                y_cen = y + 0.5 * h
                w *= ratio
                h *= ratio
                x = x_cen - w * 0.5
                y = y_cen - h * 0.5
                return np.array([x, y, w, h])

            # first create another blank image with the same size as the original image
            new_im = np.zeros(im.shape)
            if flag == 'anc':
                bb_part1 = bbox[id][0][0:4]
                bb_part2 = bbox[id][0][12:16]
            elif flag == 'pos':
                bb_part1 = bbox[id][0][4:8]
                bb_part2 = bbox[id][0][16:20]
            else:
                bb_part1 = bbox[id][0][8:12]
                bb_part2 = bbox[id][0][20:24]

            bbox1 = expand_boxes(bb_part1, ratio=1.2)
            bbox2 = expand_boxes(bb_part2, ratio=1.2)
            bb1 = np.round(bbox1).astype(np.int32).tolist()
            bb2 = np.round(bbox2).astype(np.int32).tolist()
            # print(bb)
            x1, y1, width1, height1 = boundary_check(bb1)
            x2, y2, width2, height2 = boundary_check(bb2)
            new_im[y1:y1+height1, x1:x1+width1, :] = 1.
            new_im[y2:y2+height2, x2:x2+width2, :] = 1.
            im = np.dstack((im, new_im))

            # translation and flip for image
            im = im.reshape(1, self._in_size, self._in_size, self._num_channels+1)
            x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            if nr.random() > 0.5 and self._is_train:
                trans_image = im[:, y:y+self._crop_size, x+self._crop_size:x:-1, :]
            else:
                trans_image = im[:, y:y+self._crop_size, x:x+self._crop_size, :]
            processed.append(trans_image.reshape(1, self._crop_size, self._crop_size, self._num_channels+1))
        # data = np.concatenate(processed, axis=0)
        data = np.reshape(processed, (len(imlist), self._crop_size, self._crop_size, self._num_channels+1))
        return data

    def transform_all_part(self, imlist, bbox, flag='anc'):
        processed = []
        # get the bbox for anc/pos/neg
        if len(imlist)!=len(bbox):
            print('The number of triplets is different with the number of bbox.')
            return
        bb_part1 = []
        bb_part2 = []
        if flag == 'anc':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][0:4])
                bb_part2.append(bbox[i][0][12:16])
        elif flag == 'pos':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][4:8])
                bb_part2.append(bbox[i][0][16:20])
        else:
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][8:12])
                bb_part2.append(bbox[i][0][20:24])

        processed1=[]
        processed2=[]

        def boundary_check(bbox):
            x, y, w, h = bbox
            x = min(max(0, x), 250)
            y = min(max(0, y), 250)
            w = min(max(5, w), 256-x)
            h = min(max(5, h), 256-y)
            return [x, y, w, h]

        def expand_boxes(bbox, ratio=1.2):
            x, y, w, h = bbox
            x_cen = x + 0.5 * w
            y_cen = y + 0.5 * h
            w *= ratio
            h *= ratio
            x = x_cen - w * 0.5
            y = y_cen - h * 0.5
            return np.array([x, y, w, h])

        for id, im in enumerate(imlist):
            # if im.shape[-1] == self._crop_size:
            #     processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
            #     continue
            # crop image based on bounding box and then scale
            for i in range(2):
                if i == 0:
                    bb_part = bb_part1
                else:
                    bb_part = bb_part2
                # if len(bb_part[id])<4:
                #     bb_part[id] = [1,1,self._in_size-1,self._in_size-1]
                bbox = expand_boxes(bb_part[id])
                bb = np.round(bbox).astype(np.int32).tolist()
                # print(bb)
                x, y, width, height = boundary_check(bb)
                # x = int(bb_part[id][0])
                # y = int(bb_part[id][1])
                # width = int(bb_part[id][2])
                # height = int(bb_part[id][3])
                part_image = np.require(np.squeeze(im[y:y+height, x:x+width, :], axis=2), requirements='C')
                # print(part_image.shape)
                part_image = resize(part_image, [self._crop_size, self._crop_size], preserve_range=True)
                part_image = part_image[:, :, np.newaxis]
                # plt.imshow(part_image)
                # plt.show()
                if i == 0:
                    processed1.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                else:
                    processed2.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))

        # data = np.concatenate(processed, axis=0)
        data_part1 = np.reshape(processed1, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        data_part2 = np.reshape(processed2, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data_part1, data_part2

    def transform_bbox(self, imlist, bbox, flag='anc'):
        processed = []
        # get the bbox for anc/pos/neg
        if len(imlist)!=len(bbox):
            print('The number of triplets is different with the number of bbox.')
            return
        bb_part1 = []
        bb_part2 = []
        if flag == 'anc':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][0:4])
                bb_part2.append(bbox[i][0][12:16])
        elif flag == 'pos':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][4:8])
                bb_part2.append(bbox[i][0][16:20])
        else:
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][8:12])
                bb_part2.append(bbox[i][0][20:24])

        processed1=[]
        processed2=[]

        def boundary_check(bbox):
            x, y, w, h = bbox
            x = min(max(0, x), 250)
            y = min(max(0, y), 250)
            w = min(max(5, w), 256-x)
            h = min(max(5, h), 256-y)
            return [x, y, w, h]

        def expand_boxes(bbox, ratio=1.2):
            x, y, w, h = bbox
            x_cen = x + 0.5 * w
            y_cen = y + 0.5 * h
            w *= ratio
            h *= ratio
            x = x_cen - w * 0.5
            y = y_cen - h * 0.5
            return np.array([x, y, w, h])

        for id, im in enumerate(imlist):
            # if im.shape[-1] == self._crop_size:
            #     processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
            #     continue
            # crop image based on bounding box and then scale
            for i in range(2):
                if i == 0:
                    bb_part = bb_part1
                else:
                    bb_part = bb_part2
                # if len(bb_part[id])<4:
                #     bb_part[id] = [1,1,self._in_size-1,self._in_size-1]
                bbox = expand_boxes(bb_part[id])
                bb = np.round(bbox).astype(np.int32).tolist()
                # print(bb)
                x, y, width, height = boundary_check(bb)
                # x = int(bb_part[id][0])
                # y = int(bb_part[id][1])
                # width = int(bb_part[id][2])
                # height = int(bb_part[id][3])
                part_image = np.require(np.squeeze(im[y:y+height, x:x+width, :], axis=2), requirements='C')
                # print(part_image.shape)
                part_image = resize(part_image, [self._crop_size, self._crop_size], preserve_range=True)
                part_image = part_image[:, :, np.newaxis]
                # plt.imshow(part_image)
                # plt.show()
                if i == 0:
                    processed1.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                else:
                    processed2.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))

        # data = np.concatenate(processed, axis=0)
        data_part1 = np.reshape(processed1, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        data_part2 = np.reshape(processed2, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data_part1, data_part2


class MemoryBlockManager:
    def __init__(self, data_path, has_label=False):
        self.verbose = False
        self.mean = None
        self.data_path = data_path
        self.num_samples = 0
        self.sample_id = 0
        self.batch_data = None
        self.batch_label = None
        self.has_label = has_label
        self.load_data_block()

    def pop_sample(self):
        if self.sample_id >= self.num_samples:
            self.sample_id = 0
        self.sample_id = self.sample_id + 1
        if self.has_label:
            return self.batch_data[self.sample_id-1, ::], \
                   self.batch_label[self.sample_id-1]
        else:
            return self.batch_data[self.sample_id-1, ::]

    def get_sample(self, sample_idx):
        if self.has_label:
            return self.batch_data[sample_idx, ::], self.batch_label[sample_idx]
        else:
            return self.batch_data[sample_idx, ::]

    def pop_batch(self, batch_size):
        if self.sample_id >= self.num_samples:
            return None
        prev_sample_id = self.sample_id
        self.sample_id = min(self.num_samples, self.sample_id+batch_size)
        data = self.batch_data[prev_sample_id: self.sample_id, ::]
        if self.has_label:
            labels = self.batch_label[prev_sample_id: self.sample_id]
            return data, labels
        else:
            return data, []

    def eof(self):
        return self.sample_id >= self.num_samples

    def pop_batch_circular(self, this_batch_size):
        sample_ind = self.pop_batch_inds_circular(this_batch_size)
        data = self.batch_data[sample_ind, ::]
        if self.has_label:
            labels = self.batch_label[sample_ind]
            return sample_ind, data, labels
        else:
            return sample_ind, data

    def pop_batch_inds_circular(self, this_batch_size):
        sample_ind = range(self.sample_id, self.sample_id+this_batch_size)
        sample_ind = [id % self.num_samples for id in sample_ind]
        self.sample_id = (self.sample_id + this_batch_size) % self.num_samples
        return sample_ind

    def load_data_block(self):
        print('loading data %s' % self.data_path)
        t = time()
        dic = loadmat(self.data_path)
        self.num_samples = dic['data'].shape[0]
        self.batch_data = dic['data']
        if self.has_label:
            self.batch_label = dic['labels']
        print('data loaded %s (%0.2f sec.)' % (
            self.get_data_shape_str(self.batch_data), time()-t))

    def get_data_shape_str(self, data):
        s = ''
        shape = data.shape
        for d in shape:
            s += '%dx' % d
        s = s[:-1]
        return s

    def get_num_instances(self):
        return self.num_samples


def sample_triplets_trueMatch(anc_inds, phase):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        pos_id = anc_id
        pos_inds.append(pos_id)
        # all_inds = np.unique(triplets)
        if phase == "TRAIN":
            all_inds = range(0,400)
        else:
            all_inds = range(0,168)
        neg_list = np.setdiff1d(all_inds, anc_id)
        neg_id = np.random.choice(neg_list)
        neg_inds.append(neg_id)
    return pos_inds, neg_inds


def sample_triplets_pos_neg(anc_inds, triplets, neg_list, hard_ratio):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        idx = nr.randint(len(tuples))
        pos_id, neg_id = tuples[idx]
        pos_inds.append(pos_id)
        # import pdb
        # pdb.set_trace()
        if nr.rand() > hard_ratio:  # sample easy
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
    return pos_inds, neg_inds


def read_json(fpath):
    with open(fpath) as data_file:
        data = json.load(data_file)
    return data


def read_mat(fpath):
    with open(fpath) as data_file:
        import pdb
        pdb.set_trace()
        data = loadmat(data_file)
    return data


class SMTSApi(object):
    def __init__(self, ann_path=None, dataset_root=None, name=None):
        self._dataset_root = dataset_root
        self._dbname = name
        flag_json = 1
        if flag_json == 1:
            if ann_path is None:
                ann_path = os.path.join(dataset_root, name,
                                        '%s_annotation.json' % name)
            self._annotation = read_json(ann_path)
        else:
            if ann_path is None:
                ann_path = os.path.join(dataset_root, name,
                                        '%s_annotation.mat' % name)
            self._annotation = read_mat(ann_path)

    @property
    def name(self):
        return self._dbname

    def get_triplets(self, target_set='train'):
        if self._annotation is None:
            raise Exception('annotations not loaded')
        elif target_set.lower() == 'train':
            return self._annotation['train']['triplets']
        elif target_set.lower() == 'test':
            return self._annotation['test']['triplets']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_triplets_bbox(self, target_set='train'):
        if self._annotation is None:
            raise Exception('annotations not loaded')
        elif target_set.lower() == 'train':
            return self._annotation['train']['triplets'], self._annotation['train']['bbox']
        elif target_set.lower() == 'test':
            return self._annotation['test']['triplets'], self._annotation['test']['bbox']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_triplets_mat(self, target_set='train'):
        if self._annotation is None:
            raise Exception('annotations not loaded')
        elif target_set.lower() == 'train':
            return self._annotation['pos_list'], self._annotation['neg_list_hard'], self._annotation['neg_list_easy']
        elif target_set.lower() == 'test':
            return self._annotation['test']['triplets']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_images(self, target_set='train'):
        if target_set.lower() == 'train':
            return self._annotation['train']['images']
        elif target_set.lower() == 'test':
            return self._annotation['test']['images']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_image_pathes(self, image_inds, target_set):
        images = self.get_images(target_set)
        im_root = os.path.join(self._dataset_root, self._dbname,
                               target_set, 'images')
        impathes = []
        for image_id in image_inds:
            impathes.append(os.path.join(im_root, images[image_id]))
        return impathes

    def get_image_path(self, image_id, target_set):
        if self._dataset_root is None:
            raise Exception('should pass dataset root in initialization')
        im_name = self._annotation[target_set]['images'][image_id]
        return os.path.join(self._dataset_root, '%s/%s/%s' % (target_set, 'images', im_name))

    def get_sketch_path(self, sketh_id, target_set):
        if self._dataset_root is None:
            raise Exception('should pass dataset root in initialization')
        sk_name = self._annotation[target_set]['images'][sketh_id]
        return os.path.join(self._dataset_root, '%s/%s/%s' % (target_set, 'sketches', sk_name))


def make_negative_list(triplets):
    tri_mat = np.array(triplets)
    num_images = tri_mat.shape[0]
    all_inds = np.unique(triplets)
    neg_list = []
    for i in range(num_images):
        pos_inds = np.union1d(tri_mat[i, :, 0], tri_mat[i, :, 1])
        neg_inds = np.setdiff1d(all_inds, pos_inds).reshape([1, -1])
        neg_list.append(neg_inds)
    return np.concatenate(neg_list).astype(np.int32)


def load_triplets(triplet_path, subset):
    smts_api = SMTSApi(triplet_path)
    triplets = smts_api.get_triplets(subset)
    return triplets, make_negative_list(triplets)


class TripletSamplingDataFetcher(Process):
    def __init__(self, queue, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        """Setup the TripletSamplingDataLayer."""
        super(TripletSamplingDataFetcher, self).__init__()
        #        mean = mean
        self._queue = queue
        self._phase = phase
        self.sketch_transformer = Transformer(225, 1, mean, self._phase == "TRAIN")
        self.sketch_dir = sketch_dir
        self.anc_bm = MemoryBlockManager(sketch_dir)
        self.pos_neg_bm = MemoryBlockManager(image_dir)
        self.hard_ratio = hard_ratio
        self.mini_batchsize = batch_size
        self.load_triplets(triplet_path)

    def load_triplets(self, triplet_path):
        self.triplets, self.neg_list = load_triplets(triplet_path, self._phase)

    def get_next_batch(self):
        anc_batch = []; pos_batch = []; neg_batch = []
        # sampling
        anc_inds = self.anc_bm.pop_batch_inds_circular(self.mini_batchsize)
        if 'handbags' in self.sketch_dir:
            # positive are always true match
            pos_inds, neg_inds = sample_triplets_trueMatch(anc_inds, self._phase)
        else:
            pos_inds, neg_inds = sample_triplets_pos_neg(anc_inds, self.triplets, self.neg_list, self.hard_ratio)

        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((256, 256, 1)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((256, 256, 1)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((256, 256, 1)))
        # apply transform
        anc_batch = self.sketch_transformer.transform_all(anc_batch).astype(np.uint8)
        pos_batch = self.sketch_transformer.transform_all(pos_batch).astype(np.uint8)
        neg_batch = self.sketch_transformer.transform_all(neg_batch).astype(np.uint8)
        self._queue.put((anc_batch, pos_batch, neg_batch))

    def run(self):
        print('TripletSamplingDataFetcher started')
        while True:
            self.get_next_batch()


def rand_rotate(im, rotate_amp):
    deg = (2 * nr.rand() - 1) * rotate_amp
    # print deg
    rot_im = rotate(im, deg, mode='nearest') * 255
    return rot_im.astype(np.uint8)


def main(subset, sketch_dir, image_dir, sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size, phase, phase_te, net_model):

    # ITERATIONS = 20000
    ITERATIONS = 10
    VALIDATION_TEST = 200
    perc_train = 0.9
    MARGIN = 0.3
    SAVE_STEP = 200
    model_path = "./model/%s/%s/" % (subset, net_model)
    pre_trained_model = './model/sketchnet_init.npy'
    pre_step = 0
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # Siamease place holders
    train_anchor_data = tf.compat.v1.placeholder(tf.float32, shape=(None, 225, 225, 1), name="anchor")
    train_positive_data = tf.compat.v1.placeholder(tf.float32, shape=(None, 225, 225, 1), name="positive")
    train_negative_data = tf.compat.v1.placeholder(tf.float32, shape=(None, 225, 225, 1), name="negative")
    output_data = tf.compat.v1.placeholder(tf.float32, shape=(None, 225, 225, 1))

    # Creating the architecture
    if net_model == 'deep_sbir':
        with tf.compat.v1.variable_scope('for_reuse_scope'):
            train_anchor = sketch_a_net_sbir(tf.cast(train_anchor_data, tf.float32) - mean, True)
            tf.compat.v1.get_variable_scope().reuse_variables()
            train_positive = sketch_a_net_sbir(tf.cast(train_positive_data, tf.float32) - mean, True)
            train_negative = sketch_a_net_sbir(tf.cast(train_negative_data, tf.float32) - mean, True)
            output = sketch_a_net_sbir(tf.cast(output_data, tf.float32) - mean, True)
    elif net_model == 'DSSA':
        with tf.compat.v1.variable_scope('for_reuse_scope'):
            train_anchor = sketch_a_net_dssa(tf.cast(train_anchor_data, tf.float32) - mean, True)
            tf.compat.v1.get_variable_scope().reuse_variables()
            train_positive = sketch_a_net_dssa(tf.cast(train_positive_data, tf.float32) - mean, True)
            train_negative = sketch_a_net_dssa(tf.cast(train_negative_data, tf.float32) - mean, True)
            output = sketch_a_net_dssa(tf.cast(output_data, tf.float32) - mean, True)
    else:
        print('Please define the net_model')

    init_ops = init_variables()
    loss, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, MARGIN)

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = 0.001
    data_sampler = TripletSamplingLayer()
    data_sampler_te = TripletSamplingLayer()
    data_sampler.setup(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
    data_sampler_te.setup(sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size, phase_te)
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,
                                                                                                       global_step=batch)
    #validation_prediction = tf.nn.softmax(lenet_validation)
    # saver = tf.compat.v1.train.Saver(max_to_keep=1)
    dst_path = './model'
    model_id = '%s/%s/log.txt' % (subset, net_model)
    filename = dst_path+'/'+model_id
    # f = open(filename, 'a')
    # Training
    with tf.compat.v1.Session() as session:

        session.run(tf.compat.v1.global_variables_initializer())
        session.run(init_ops)
        for step in range(ITERATIONS):
            f = open(filename, 'a')
            batch_anchor, batch_positive, batch_negative = data_sampler.get_next_batch()
            feed_dict = {train_anchor_data: batch_anchor,
                         train_positive_data: batch_positive,
                         train_negative_data: batch_negative
                         }
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            print("Iter %d: Loss Train %f" % (step+pre_step, l))
            f.write("Iter "+str(step+pre_step) + ": Loss Train: " + str(l))
            f.write("\n")
            # train_writer.add_summary(summary, step)

            if step % SAVE_STEP == 0:
                str_temp = '%smodel-iter%d.npy' % (model_path, step+pre_step)
                save_dict = {var.name: var.eval(session) for var in tf.compat.v1.global_variables()}
                np.save(str_temp, save_dict)

            if step % VALIDATION_TEST == 0:
                batch_anchor, batch_positive, batch_negative = data_sampler_te.get_next_batch()

                feed_dict = {train_anchor_data: batch_anchor,
                             train_positive_data: batch_positive,
                             train_negative_data: batch_negative
                             }

                lv = session.run([loss], feed_dict=feed_dict)
                # test_writer.add_summary(summary, step)
                print("Loss Validation {0}".format(lv))
                f.write("Loss Validation: " + str(lv))
                f.write("\n")
            f.close()
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./model/shoes/models')
        builder.add_meta_graph_and_variables(session, ['deep_sbir'])
        builder.save()

if __name__ == '__main__':
    # 'deep_sbir'(the model of cvpr16) or 'DSSA'(the model of iccv17)
    net_model = 'deep_sbir'  
    subset = 'shoes'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 128
    phase = 'TRAIN'
    phase_te = 'TEST'
    base_path = './data'
    sketch_dir = '%s/%s/%s_sketch_db_%s.mat' % (base_path, subset, subset, phase.lower())
    image_dir = '%s/%s/%s_edge_db_%s.mat' % (base_path, subset, subset, phase.lower())
    triplet_path = '%s/%s/%s_annotation.json' % (base_path, subset, subset) # pseudo annotations for handbags
    sketch_dir_te = '%s/%s/%s_sketch_db_%s.mat' % (base_path, subset, subset, phase_te.lower())
    image_dir_te = '%s/%s/%s_edge_db_%s.mat' % (base_path, subset, subset, phase_te.lower())
    main(subset, sketch_dir, image_dir, sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size, phase, phase_te, net_model)
