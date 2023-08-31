# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import argparse
from unet import UNet
from utils import compile_frames_to_gif

"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--embedding_ids', default='embedding_ids', type=str, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--interpolate', dest='interpolate', type=int, default=0,
                    help='interpolate between different embedding vectors')
parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
parser.add_argument('--uroboros', dest='uroboros', type=int, default=0,
                    help='Shōnen yo, you have stepped into uncharted territory')
# args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    path_model = '22_8_5/checkpoint/experiment_60_batch_16'
    path_data_obj = 'test_seench_unseenfont_obj'
    path_save = path_data_obj


    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            model = UNet(batch_size=16)
            model.register_session(sess)
            model.build_model(is_training=True)
            model.infer(model_dir=path_model,
            source_obj=path_data_obj,save_dir=path_save)
                # if args.output_gif:
                #     gif_path = os.path.join(args.save_dir, args.output_gif)
                #     compile_frames_to_gif(args.save_dir, gif_path)
                #     print("gif saved at %s" % gif_path)

        

if __name__ == '__main__':
    tf.app.run()
