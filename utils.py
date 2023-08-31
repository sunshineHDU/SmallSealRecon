# -*- coding: utf-8 -*-
import os
import glob

import imageio
import scipy.misc as misc
import scipy.misc
import numpy as np
from io import StringIO, BytesIO
from PIL import Image
import cv2
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = imageio.imread(img).astype(np.float)
    # todo 2--->3
    side = int(mat.shape[1] / 3)
    assert side * 3 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:side*2]  # source
    img_C = mat[:, side*2:]  # source

    return img_A, img_B, img_C


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    enlarged = misc.imresize(img, [nw, nh])
    # enlarged = np.array(Image.fromarray(img).resize((nw, nh)))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(misc.imread(f), interp='nearest', size=0.33) for f in frames]

    # images = [cv2.resize(imageio.imread(f), (len(f)//3, len(f[0]) // 3), interpolation='INTER_NEAREST')
    #               for f in frames]
    misc.imsave(gif_file, images, duration=0.1)
    return gif_file

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y)
    return acc

def accuracy_num(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    return corr


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta