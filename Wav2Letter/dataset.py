import glob
import io
import logging

import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm

from . import data


ENGLISH_CHAR_MAP = [
    '',
    '_',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', ':', '(', ')', '.', ',', '/', '$',
    "'",
    " "
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


class ImageCommand(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.max_width = 150
        self.max_target_length = 80
        self.max_target_length_1 = self.max_target_length - 1
        _, inv_charset = read_charset()
        self.labels = []

        self.inv_charset = inv_charset
        self.intencode = data.IntegerEncode()

    def _emdeding(self, label):
        label = str(label, encoding='UTF-8')
        labels = []
        for c in label.lower():
            if c == '_':
                continue
            v = self.inv_charset.get(c, -1)
            if v > 0:
                labels.append(v)
        if len(labels) < 1:
            labels.append(self.inv_charset[' '])
        if len(labels) > self.max_target_length_1:
            labels = labels[:self.max_target_length_1]
        labels.append(1)
        return np.array(labels, dtype=np.int64)

    def load_vectors(self, progress_bar=True):
        datasets_files = []
        for tf_file in glob.iglob(self.data_path + '/*.tfrecord'):
            datasets_files.append(tf_file)

        inputs = []

        pg = tqdm.tqdm if progress_bar else lambda x: x

        for tf_file in datasets_files:
            tf_iter = tf.python_io.tf_record_iterator(tf_file)
            # with open(tf_file, 'rb') as f:
            #     raw_data = f.read()

            for raw_data in pg(tf_iter):
                res = tf.train.Example()
                res.ParseFromString(raw_data)

                encoded_image = res.features.feature['image/encoded'].bytes_list.value[0]
                width = res.features.feature['width'].int64_list.value[0]
                height = res.features.feature['height'].int64_list.value[0]
                text = res.features.feature['image/text'].bytes_list.value[0]
                img = Image.open(io.BytesIO(encoded_image)).convert('RGB')

                w = np.maximum(float(width), 1.0)
                h = np.maximum(float(height), 1.0)

                ratio_w = np.maximum(w / self.max_width, 1.0)
                ratio_h = np.maximum(h / 32.0, 1.0)
                ratio = np.maximum(ratio_w, ratio_h)

                nw = np.maximum(np.floor_divide(w, ratio), 1.0).astype(np.int32)
                nh = np.maximum(np.floor_divide(h, ratio), 1.0).astype(np.int32)

                img = img.resize((nh, nw))

                padw = np.maximum(0, int(self.max_width) - nw)
                padh = np.maximum(0, 32 - nh)
                padded = Image.new(
                    'RGB',
                    (nh + padh, nw + padw),  # A4 at 72dpi
                    (0, 0, 0),
                )  # Black
                padded.paste(img, (0, 0))  # Not centered, top-left corner

                img = np.array(padded).astype(np.float32) / 127.5 - 1
                img = img.reshape([-1, 3])

                # label = self._emdeding(text)
                logging.info('label out {}'.format(text.decode()))
                inputs.append(img)
                self.labels.append(text.decode())

                # labels.append(label)
            break

        # Preprocess labels
        targets = []
        for l in self.labels:
            target = self.intencode.convert_to_ints(l)
            targets.append(target)

        return np.stack(inputs), np.stack(targets)


def tf_input_fn(params, is_training):
    max_width = params['max_width']
    batch_size = params['batch_size']
    _, inv_charset = read_charset()
    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.tfrecord'):
        datasets_files.append(tf_file)
    max_target_seq_length = params['max_target_seq_length']
    max_target_seq_length_1 = max_target_seq_length - 1

    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)

        def _emdeding(label):
            label = str(label, encoding='UTF-8')
            labels = []
            for c in label.lower():
                if c == '_':
                    continue
                v = inv_charset.get(c, -1)
                if v > 0:
                    labels.append(v)
            if len(labels) < 1:
                labels.append(inv_charset[' '])
            if len(labels) > max_target_seq_length_1:
                labels = labels[:max_target_seq_length_1]
            labels.append(1)
            return np.array(labels, dtype=np.int64)

        def _parser(example):
            zero = tf.zeros([1], dtype=tf.int64)
            features = {
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/height':
                    tf.FixedLenFeature([1], tf.int64, default_value=zero),
                'image/width':
                    tf.FixedLenFeature([1], tf.int64, default_value=zero),
                'image/text':
                    tf.FixedLenFeature((), tf.string, default_value=''),
            }
            res = tf.parse_single_example(example, features)
            img = tf.image.decode_png(res['image/encoded'], channels=3)
            original_w = tf.cast(res['image/width'][0], tf.int32)
            original_h = tf.cast(res['image/height'][0], tf.int32)
            img = tf.reshape(img, [original_h, original_w, 3])
            w = tf.maximum(tf.cast(original_w, tf.float32), 1.0)
            h = tf.maximum(tf.cast(original_h, tf.float32), 1.0)
            ratio_w = tf.maximum(w / max_width, 1.0)
            ratio_h = tf.maximum(h / 32.0, 1.0)
            ratio = tf.maximum(ratio_w, ratio_h)
            nw = tf.cast(tf.maximum(tf.floor_div(w, ratio), 1.0), tf.int32)
            nh = tf.cast(tf.maximum(tf.floor_div(h, ratio), 1.0), tf.int32)
            img = tf.image.resize_images(img, [nh, nw])
            padw = tf.maximum(0, int(max_width) - nw)
            padh = tf.maximum(0, 32 - nh)
            img = tf.image.pad_to_bounding_box(img, 0, 0, nh + padh, nw + padw)
            img = tf.cast(img, tf.float32) / 127.5 - 1
            label = res['image/text']
            label = tf.py_func(_emdeding, [label], tf.int64)
            logging.info('label out {}'.format(label))
            return img, label

        ds = ds.map(_parser)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(1000))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]), padding_values=(0.0, np.int64(1)))
        return ds

    return _input_fn
