import io
import datetime
import logging
import numpy as np
import pickle
import unittest

import cProfile
import pstats
from pstats import SortKey

from model_utils import ModelUtils
from rnn_using_only_numpy import RnnWithNumpy

logger = logging.getLogger(__name__)


# Main training parameters
dim_hidden = 128
fixed_learning_rate = 0.004
sequence_length = 25
batch_size = 20
max_epoch = 3000
text_generation_prompt = 'ROMEO:'
text_file = './training_data/shakespeare.txt'

shorted_text = (
'ROMEO: Is the day so young?''')

def save_model(model):
    filepath_template = './saved_numpy_models/model_%s.pkl'
    filepath = filepath_template % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def model_training_batch_callback(model, prompt, char_to_id_map, id_to_char_map, output_length=100):
    generated_text = ModelUtils.generate_text(model, prompt, char_to_id_map, id_to_char_map, output_length=100)
    logger.info("Generated text=%s", generated_text)
    save_model(model)


def test_simple_training():
    vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data(
        filepath=text_file, sequence_length=sequence_length)
    dim_vocab = len(vocab)

    def _model_batch_callback(model):
        model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

    rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=dim_hidden)

    logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, fixed_learning_rate=%f",
        dim_vocab, dim_hidden, sequence_length, fixed_learning_rate)

    rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=fixed_learning_rate,
        batch_size=batch_size, max_epoch=max_epoch, batch_callback=_model_batch_callback)


def profile_simple_training():
    vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data_from_text(
        text=shorted_text, sequence_length=sequence_length)

    dim_vocab = len(vocab)

    def _model_batch_callback(model):
        model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

    rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=dim_hidden)

    logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, fixed_learning_rate=%f",
        dim_vocab, dim_hidden, sequence_length, fixed_learning_rate)

    rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=fixed_learning_rate,
        batch_size=1, max_epoch=4, batch_callback=_model_batch_callback)


class TestNumpyRnnTextGeneration(unittest.TestCase):

    def test_simple_training(self):
        test_simple_training()


    def test_with_very_short_training_data(self):
        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data_from_text(
            text=shorted_text, sequence_length=sequence_length)

        dim_vocab = len(vocab)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=dim_hidden)

        logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, fixed_learning_rate=%f",
            dim_vocab, dim_hidden, sequence_length, fixed_learning_rate)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=fixed_learning_rate,
            batch_size=batch_size, max_epoch=max_epoch, batch_callback=_model_batch_callback)


    def test_continue_with_previous_model_short_training_data(self):
        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data_from_text(
            text=shorted_text, sequence_length=sequence_length)
        dim_vocab = len(vocab)

        filepath = './saved_numpy_models/model_2023_04_07_19_28_21.pkl'

        with open(filepath, 'rb') as file:
            rnn_model = pickle.load(file)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        learning_rate = 0.02
        logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, fixed_learning_rate=%f",
            dim_vocab, dim_hidden, sequence_length, learning_rate)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=0.001,
            batch_size=batch_size, max_epoch=max_epoch, batch_callback=_model_batch_callback)


    def test_continue_with_previous_model(self):
        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data(
            filepath=text_file, sequence_length=sequence_length)
        dim_vocab = len(vocab)

        filepath = './saved_numpy_models/model_2023_04_07_14_12_13.pkl'

        with open(filepath, 'rb') as file:
            rnn_model = pickle.load(file)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        learning_rate = 0.001
        logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, fixed_learning_rate=%f",
            dim_vocab, dim_hidden, sequence_length, learning_rate)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=0.001,
            batch_size=batch_size, max_epoch=max_epoch, batch_callback=_model_batch_callback)

    def profile_simple_training(self):
        # Creating profile object
        ob = cProfile.Profile()
        ob.enable()

        cProfile.run('profile_simple_training()')

        sec = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
        ps.print_stats()


    def test_simple_training_longer(self):
        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = numpy_rnn_text_generation.prepare_data_from_text(test_text, 40)

        training_data_truncation = -1
        rnn_model = rnn_using_only_np.Model(len(id_to_char_map), 256, debug_output=False)

        def training_batch_callback():
            save_model(rnn_model)
            generated_text = ''.join(numpy_rnn_text_generation.generate_text(rnn_model, u'洪七公脸如白纸', char_to_id_map, id_to_char_map, output_length=100))
            logger.info('generated_text=' + generated_text)

        rnn_model.train(input_id_seqs[:training_data_truncation], label_id_seqs[:training_data_truncation], learning_rate = 0.003, nepoch=50,
            batch_size=len(input_id_seqs), callback_per_batch=training_batch_callback)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()