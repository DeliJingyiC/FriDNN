import datetime
import sys
import time
import csv
from argparse import ArgumentParser
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.optimizers import Adam
from keras.models import load_model

from helpers import sorted_alphanumeric
from lists import core_test_set_speakers, fricative_list, val_audio_list, fricative_dict, fricative_dict_rev
from model import FriDNN
from test_utils import prediction_whole_core_test, calculate_performance
from train_utils import DataGenerator

from pathlib import Path

from test_utils import process_audio

import matplotlib.pyplot as plt

from tqdm import trange

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--audio_dir',
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

    parser.add_argument('--window_size',
                        type=int,
                        help='window size of the sounds')
    parser.add_argument(
        '--job_id',
        type=str,
    )

    args = parser.parse_args()
    args.audio_dir = Path(args.audio_dir)
    args.model_dir = Path(args.model_dir)
    args.output_dir = Path(args.output_dir) / args.job_id

    ck_dir = args.model_dir / 'checkpoints'

    different_epoch_paths = ck_dir.iterdir()
    different_epoch_paths = sorted_alphanumeric(different_epoch_paths)

    last_checkpoint_path = different_epoch_paths[-1]

    model = load_model(last_checkpoint_path)

    input_files = [
        f for f in args.audio_dir.iterdir() if (f.name.endswith("wav"))
    ]
    s_output_dir = args.output_dir / f"s_output"
    s_output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / f"figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    name = []
    carry_s = []
    for i in trange(len(input_files)):
        audio_path = input_files[i]
        print(audio_path)
        loaded_utterance, input_to_prediction = process_audio(
            audio_path,
            args.delay,
            args.window_size,
        )
        ind = np.arange(len(input_to_prediction))
        prediction = model.predict(input_to_prediction, batch_size=32)
        prediction = prediction.squeeze()

        total_prediction_binary = np.argmax(prediction, axis=1)
        name.append(audio_path.name)

        carry_s.append(
            len(total_prediction_binary[total_prediction_binary == 0]))
        """
        plt.figure()

        for i in range(0, len(fricative_dict)):
            plt.subplot(len(fricative_dict), 1, i + 1)

            plt.plot(
                ind,
                loaded_utterance,
            )
            local_ind = total_prediction_binary == i
            plt.plot(
                ind[local_ind],
                loaded_utterance[local_ind],
                label=fricative_dict_rev[i],
                linestyle="",
                marker="*",
            )
            plt.legend()
        plt.savefig(
            figure_dir / f"{audio_path.name.replace('.wav', '.png')}",
            dpi=600,
        )
        plt.close()
        """
    dataframe = pd.DataFrame({'soundfile': name, 's': carry_s})
    dataframe.to_csv(s_output_dir / "s_output.csv", index=False, sep=',')
    #print(prediction.shape)
