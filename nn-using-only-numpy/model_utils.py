'''
A Utility class to do the peripheral jobs supporting the machine learnings works,
such as training data preparation, character/id mapping, sequence prediction.
'''

import datetime
import logging
import math
import numpy as np
import os
import pickle
import time

import rnn_using_only_numpy


logger = logging.getLogger(__name__)


class ModelUtils:
    @staticmethod
    def prepare_data(filepath, sequence_length, truncation=-1, shuffle_data=False, percentage_val_set=-1):
        with open(filepath, 'rb') as f:
            text = f.read()
        text = text.decode('utf-8')
        text = text[:truncation]

        return ModelUtils.prepare_data_from_text(text, sequence_length, shuffle_data, percentage_val_set)


    @staticmethod
    def save_model(model, char_to_id_map, id_to_char_map, description='No Description Provided'):
        filepath_template = './saved_numpy_models/model_%s.pkl'
        filepath = filepath_template % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if not os.path.isdir("./saved_numpy_models"):
            os.makedirs("./saved_numpy_models")

        saved_data = {
            'model': model,
            'char_to_id_map': char_to_id_map,
            'id_to_char_map': id_to_char_map,
            'description': description,
        }

        with open(filepath, 'wb') as file:
            pickle.dump(saved_data, file)


    def load_model_and_maps(filepath):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        (model, char_to_id_map, id_to_char_map, description) = (
            data['model'],
            data['char_to_id_map'],
            data['id_to_char_map'],
            data['description'])

        return (model, char_to_id_map, id_to_char_map, description)


    @staticmethod
    def prepare_data_from_text(text, sequence_length, shuffle_data=False, percentage_val_set=-1):
        # length of text is the number of characters in it
        logger.info(f'Length of text: {len(text)} characters')

        vocab = sorted(set(text))
        char_to_id_map, id_to_char_map = ModelUtils.build_dict(vocab)
        vocab = sorted(set(char_to_id_map.keys()))

        logger.info('Started converting creating input/label sequences.')
        input_seqs, label_seqs = ModelUtils.text_to_list_of_sequences_split(text, sequence_length)
        total_sample_size = len(input_seqs)

        input_id_seqs = ModelUtils.string_sequences_to_id_sequences(input_seqs, char_to_id_map)
        label_id_seqs = ModelUtils.string_sequences_to_id_sequences(label_seqs, char_to_id_map)
        validation_input_id_seqs = None
        validation_label_id_seqs = None

        if shuffle_data:
            input_id_seqs, label_id_seqs = ModelUtils.shuffle_training_data(input_id_seqs, label_id_seqs)

        if percentage_val_set > 0:
            pivot_index = int(len(input_id_seqs) * (1 - percentage_val_set))
            validation_input_id_seqs = input_id_seqs[pivot_index:]
            validation_label_id_seqs = label_id_seqs[pivot_index:]
            input_id_seqs = input_id_seqs[:pivot_index]
            label_id_seqs = label_id_seqs[:pivot_index]

        logger.info('Done converting creating input/label sequences. Total sample size=%d, Training sample size=%d, vocab size=%d, sequence length=%d.',
            total_sample_size, len(input_id_seqs), len(vocab), sequence_length)

        return vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs, validation_input_id_seqs, validation_label_id_seqs


    # In case of text generation, just use a fixed length for the training sequence
    @staticmethod
    def text_to_list_of_sequences(text, sequence_length):
        input_seqs = []
        label_seqs = []

        for start_index in range(len(text) - sequence_length - 1):
            input_seqs.append(text[start_index: start_index + sequence_length])
            label_seqs.append(text[start_index + 1: start_index + sequence_length + 1])

        return input_seqs, label_seqs


    # This is the rolling version.
    @staticmethod
    def text_to_list_of_sequences(text, sequence_length):
        input_seqs = []
        label_seqs = []

        for start_index in range(len(text) - sequence_length - 1):
            input_seqs.append(text[start_index: start_index + sequence_length])
            label_seqs.append(text[start_index + 1: start_index + sequence_length + 1])

        return input_seqs, label_seqs


    # Split verison
    @staticmethod
    def text_to_list_of_sequences_split(text, sequence_length):
        input_seqs = []
        label_seqs = []

        start_index = 0

        # Drop out the last sequence which will not match length for input and output.
        while start_index + sequence_length + 1 < len(text):
            input_seqs.append(text[start_index: start_index + sequence_length])
            label_seqs.append(text[start_index + 1: start_index + sequence_length + 1])

            start_index += sequence_length

        return input_seqs, label_seqs


    @staticmethod
    def string_sequences_to_id_sequences(string_sequences, char_to_id_map):
        id_sequences = []
        for sequence in string_sequences:
            id_sequences.append(ModelUtils.char_array_to_id_array(sequence, char_to_id_map))

        return id_sequences


    @staticmethod
    def char_array_to_id_array(char_array, char_to_id_map):
        # TODO, low-pri: maybe do a type check here?

        id_array = []
        for char in char_array:
            id_array.append(char_to_id_map[char])

        return id_array

    @staticmethod
    def id_array_to_char_array(id_array, id_to_char_map):
        # TODO, low-pri: maybe do a type check here?

        char_array = []
        for id in id_array:
            char_array.append(id_to_char_map[id])

        return char_array


    @staticmethod
    def build_dict(vocab_set):
        # index 0 reserved for ['UNK']
        id_to_char_map = dict((i + 1, s) for i, s in enumerate(vocab_set))
        id_to_char_map[0] = '[UNK]'

        char_to_id_map = dict((s, i) for i, s in id_to_char_map.items())

        return char_to_id_map, id_to_char_map


    @staticmethod
    def generate_text(model, prompt, char_to_id_map, id_to_char_map, output_length=100):
        logger.info("Generating Text...")

        prompt_input_ids = ModelUtils.char_array_to_id_array(prompt, char_to_id_map)
        result_ids = prompt_input_ids.copy()

        model.reset_prev_state()

        for input_id in prompt_input_ids:
            output_id = model.forward_and_predict_carry_state(input_id)
            # Discard all output_ids except the last one.

        result_ids.append(output_id)
        for i in range(output_length):
            output_id = model.forward_and_predict_carry_state(output_id)
            result_ids.append(output_id)

        results = ModelUtils.id_array_to_char_array(result_ids, id_to_char_map)
        results = ''.join(results)
        return results


    @staticmethod
    def shuffle_training_data(input_id_seqs, label_id_seqs):
        assert len(input_id_seqs) == len(label_id_seqs)
        logger.info('Started shuffling training data.')

        shuffled_input_id_seqs = []
        shuffled_label_id_seqs = []

        indices = list(range(len(input_id_seqs)))
        np.random.shuffle(indices)

        for i in indices:
            shuffled_input_id_seqs.append(input_id_seqs[i])
            shuffled_label_id_seqs.append(label_id_seqs[i])

        logger.info('Done shuffling training data.')

        return shuffled_input_id_seqs, shuffled_label_id_seqs

