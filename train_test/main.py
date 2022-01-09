import datetime
import sys
import time
from argparse import ArgumentParser
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.optimizers import Adam

from helpers import sorted_alphanumeric
from lists import core_test_set_speakers, fricative_list, val_audio_list
from model import FriDNN
from test_utils import prediction_whole_core_test, calculate_performance
from train_utils import DataGenerator

from pathlib import Path

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

# training parameters

BATCH_SIZE = 256  # bath size
NUMBER_OF_AUDIOS = 16  # number of utterance selected for a single batch
EPOCHS = 1000
WINDOW_SIZE = 320  # input size of the network


def train_and_test_model(path_to_TIMIT, pre_delay, model_dir, output_dir,
                         only_test):

    # creating relatives paths for test data set, validation data set and train data set
    test_audio_list = []
    test_dir = path_to_TIMIT / "TIMIT" / "TEST"
    for dialect in test_dir.iterdir():
        for person in dialect.iterdir():
            if person.name in core_test_set_speakers:
                temp_list = person.iterdir()

                temp_list = [
                    x for x in temp_list
                    if x.name.endswith('WAV') and ('SA' not in x.name)
                ]
                for sentence in temp_list:
                    test_audio_list.append(sentence)

    val_audio_full_paths = [test_dir / item for item in val_audio_list]

    train_audio_list = []
    train_dir = path_to_TIMIT / "TIMIT" / "TRAIN"
    for dialect in train_dir.iterdir():
        for person in dialect.iterdir():
            temp_list = person.iterdir()
            temp_list = [
                x for x in temp_list
                if x.name.endswith('WAV') and ('SA' not in x.name)
            ]

            for sentence in temp_list:
                train_audio_list.append(sentence)

    assert WINDOW_SIZE / pre_delay == 2, "Window size and prediction delay are set wrong!"

    ck_dir = model_dir / 'checkpoints'
    ck_dir.mkdir(parents=True, exist_ok=True)

    if not only_test:

        model_dir.mkdir(parents=True, exist_ok=True)

        # Network Creation
        name = f"{model_dir.name}-{pre_delay}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        model = FriDNN(input_size=WINDOW_SIZE)

        model.summary()

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        with open(model_dir / f"{name}_summary.txt", 'w') as f:
            with redirect_stdout(f):
                model.summary()
        log_dir = model_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        tensorboard = TensorBoard(log_dir=log_dir)
        es = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           mode='min',
                           verbose=2,
                           patience=40)

        lr = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.5,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               cooldown=0,
                               min_lr=0)

        filepath = ck_dir / "model_epoch_{epoch:02d}_valloss_{val_loss:.3f}.hdf5"

        mc1 = ModelCheckpoint(
            str(filepath),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1,
        )

        # Initialization of data generators for training and validation
        train_gen = DataGenerator(train_audio_list, BATCH_SIZE,
                                  NUMBER_OF_AUDIOS, WINDOW_SIZE, pre_delay,
                                  'training')

        val_gen = DataGenerator(val_audio_full_paths, BATCH_SIZE,
                                NUMBER_OF_AUDIOS, WINDOW_SIZE, pre_delay,
                                'validation')

        hist = model.fit_generator(
            train_gen,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[tensorboard, es, mc1, lr],
            validation_data=val_gen,
            max_queue_size=4,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=0,
        )

    different_epoch_paths = ck_dir.iterdir()
    if not only_test:
        different_epoch_paths = list(
            filter(
                lambda x_: x_.name[-4:] == "hdf5" and x_.name.split('_epoch_')[
                    0] == name.split('_epoch_')[0], different_epoch_paths))

    different_epoch_paths = sorted_alphanumeric(different_epoch_paths)

    last_checkpoint_path = different_epoch_paths[-1]

    prediction, phoneme_ground_truth = prediction_whole_core_test(
        output_dir, test_audio_list, last_checkpoint_path, WINDOW_SIZE,
        pre_delay)

    calculate_performance(output_dir, prediction, phoneme_ground_truth)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--timit_directory',
                        type=str,
                        required=True,
                        help='Directory containing TIMIT dataset.',
                        metavar='<TimitDirectory>')

    parser.add_argument('--delay',
                        type=int,
                        help='Detection delay in samples',
                        metavar='<Delay>',
                        default=160)

    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='Target directory for the experiment',
                        metavar='<TargetDirectory>')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Target directory for the experiment',
                        metavar='<TargetDirectory>')
    parser.add_argument('--test_only',
                        default=False,
                        action='store_true',
                        help='Test already trained model')

    parser.add_argument(
        '--job_id',
        type=str,
    )

    args = parser.parse_args()
    args.timit_directory = Path(args.timit_directory)
    args.model_dir = Path(args.model_dir)
    args.output_dir = Path(args.output_dir) / args.job_id

    train_and_test_model(
        args.timit_directory,
        args.delay,
        args.model_dir,
        args.output_dir,
        args.test_only,
    )
