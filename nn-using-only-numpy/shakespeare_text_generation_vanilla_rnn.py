import io
import datetime
import logging
import numpy as np
import os
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
learning_rate = 0.004
sequence_length = 70
batch_size = 15
max_epoch = 3000
text_generation_prompt = 'ROMEO:'
text_file = './training_data/shakespeare.txt'

short_text = (
'ROMEO: Is the day so young?''')

longer_text =(
'''
QUEEN MARGARET:
Thy woes will make them sharp, and pierce like mine.

DUCHESS OF YORK:
Why should calamity be full of words?

QUEEN ELIZABETH:
Windy attorneys to their client woes,
Airy succeeders of intestate joys,
Poor breathing orators of miseries!
Let them have scope: though what they do impart
Help not all, yet do they ease the heart.

DUCHESS OF YORK:
If so, then be not tongue-tied: go with me.
And in the breath of bitter words let's smother
My damned son, which thy two sweet sons smother'd.
I hear his drum: be copious in exclaims.

KING RICHARD III:
Who intercepts my expedition?

DUCHESS OF YORK:
O, she that might have intercepted thee,
By strangling thee in her accursed womb
From all the slaughters, wretch, that thou hast done!

QUEEN ELIZABETH:
Hidest thou that forehead with a golden crown,
Where should be graven, if that right were right,
The slaughter of the prince that owed that crown,
And the dire death of my two sons and brothers?
Tell me, thou villain slave, where are my children?

DUCHESS OF YORK:
Thou toad, thou toad, where is thy brother Clarence?
And little Ned Plantagenet, his son?

QUEEN ELIZABETH:
Where is kind Hastings, Rivers, Vaughan, Grey?

KING RICHARD III:
A flourish, trumpets! strike alarum, drums!
Let not the heavens hear these tell-tale women
Rail on the Lord's enointed: strike, I say!
Either be patient, and entreat me fair,
Or with the clamorous report of war
Thus will I drown your exclamations.

DUCHESS OF YORK:
Art thou my son?

KING RICHARD III:
Ay, I thank God, my father, and yourself.

DUCHESS OF YORK:
Then patiently hear my impatience.

KING RICHARD III:
Madam, I have a touch of your condition,
Which cannot brook the accent of reproof.

DUCHESS OF YORK:
O, let me speak!

KING RICHARD III:
Do then: but I'll not hear.

DUCHESS OF YORK:
I will be mild and gentle in my speech.

KING RICHARD III:
And brief, good mother; for I am in haste.

DUCHESS OF YORK:
Art thou so hasty? I have stay'd for thee,
God knows, in anguish, pain and agony.

KING RICHARD III:
And came I not at last to comfort you?

DUCHESS OF YORK:
No, by the holy rood, thou know'st it well,
Thou camest on earth to make the earth my hell.
A grievous burthen was thy birth to me;
Tetchy and wayward was thy infancy;
Thy school-days frightful, desperate, wild, and furious,
Thy prime of manhood daring, bold, and venturous,
Thy age confirm'd, proud, subdued, bloody,
treacherous,
More mild, but yet more harmful, kind in hatred:
What comfortable hour canst thou name,
That ever graced me in thy company?

KING RICHARD III:
Faith, none, but Humphrey Hour, that call'd
your grace
To breakfast once forth of my company.
If I be so disgracious in your sight,
Let me march on, and not offend your grace.
Strike the drum.

DUCHESS OF YORK:
I prithee, hear me speak.

KING RICHARD III:
You speak too bitterly.

DUCHESS OF YORK:
Hear me a word;
For I shall never speak to thee again.

KING RICHARD III:
So.
''')

def save_model(model):
    filepath_template = './saved_numpy_models/model_%s.pkl'
    folder = './saved_numpy_models'
    filepath = filepath_template % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def model_training_batch_callback(model, prompt, char_to_id_map, id_to_char_map, output_length=100):
    generated_text = ModelUtils.generate_text_beam_search(model, prompt, char_to_id_map, id_to_char_map, output_length=100)
    logger.info("Generated text=\n%s", generated_text)
    ModelUtils.save_model(model, char_to_id_map, id_to_char_map)


def profile_training():
    raise NotImplementedError('Needs to rewrite this code to accommodate latest train() signatures.')
    vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data_from_text(
        text=longer_text, sequence_length=sequence_length)

    dim_vocab = len(vocab)

    def _model_batch_callback(model):
        model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

    rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=128)

    rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, learning_rate=learning_rate,
        batch_size=20, max_epoch=4, batch_callback=_model_batch_callback)


class TestNumpyRnnTextGeneration(unittest.TestCase):

    def test_text_file_training_truncation_45000(self):
        training_parameters = {
            'thread_worker_count': 10,
            'gradient_clipping_radius': 1,
            'bptt_truncation_length': 10,
            'base_learning_rate': 0.05,
            'min_learning_rate': 0.01,
            'mini_batch_size': 10,
            'max_epoch': 200,

            # Learning Rate adjustments
            'learning_rate_reduction_ratio_when_plataeu': .9,
            'loss_plataeu_check_window': 20,
            'min_batches_since_last_lr_adjustment': 40,
            # If loss_moving_avg([-N:]) >= moving_avg([-2N: -N]) * is_plataeu_criteria_ratio, regard this as hitting a plataeu.
            'is_plataeu_criteria_ratio': 1.0
        }

        text_generation_prompt = 'MARCIUS'
        sequence_length = 100
        hidden_dim = 80

        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs, validation_inputs, validation_labels = ModelUtils.prepare_data(
            filepath=text_file, sequence_length=40, truncation=-1, shuffle_data=True, percentage_val_set=.03)

        dim_vocab = len(vocab)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=hidden_dim)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs,
            training_parameters=training_parameters, batch_callback=_model_batch_callback,
            validation_x_input_int_list_of_sequences=validation_inputs, validation_y_label_int_list_of_sequences=validation_labels)


    def test_with_very_short_training_data(self):
        training_parameters = {
            'thread_worker_count': 10,
            'gradient_clipping_radius': 1,
            'bptt_truncation_length': 10,
            'base_learning_rate': 0.003,
            'mini_batch_size': 1,
            'max_epoch': 200,

            # Learning Rate adjustments
            'learning_rate_reduction_ratio_when_plataeu': .5,
            'loss_plataeu_check_window': 20,
            'min_batches_since_last_lr_adjustment': 40,
            # If loss_moving_avg([-N:]) >= moving_avg([-2N: -N]) * is_plataeu_criteria_ratio, regard this as hitting a plataeu.
            'is_plataeu_criteria_ratio': 1.0
        }

        text_generation_prompt = 'ROMEO'
        sequence_length = 25

        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs, validation_inputs, validation_labels = ModelUtils.prepare_data_from_text(
            text=short_text, sequence_length=sequence_length, shuffle_data=True, percentage_val_set=-1)

        dim_vocab = len(vocab)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=32)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs,
            training_parameters=training_parameters, batch_callback=_model_batch_callback,
            validation_x_input_int_list_of_sequences=validation_inputs, validation_y_label_int_list_of_sequences=validation_labels)


    def test_longer_text_training(self):
        training_parameters = {
            'thread_worker_count': 2,
            'gradient_clipping_radius': 1,
            'bptt_truncation_length': 40,
            'base_learning_rate': 0.005,
            'mini_batch_size': 10,
            'max_epoch': 200,

            # Learning Rate adjustments
            'learning_rate_reduction_ratio_when_plataeu': .6,
            'loss_plataeu_check_window': 10,
            'min_batches_since_last_lr_adjustment': 8,
            # If loss_moving_avg([-N:]) >= moving_avg([-2N: -N]) * is_plataeu_criteria_ratio, regard this as hitting a plataeu.
            'is_plataeu_criteria_ratio': .99
        }
        text_generation_prompt = 'QUEEN'
        sequence_length = 20
        hidden_dim = 64

        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs, validation_inputs, validation_labels  = ModelUtils.prepare_data_from_text(
            text=longer_text, sequence_length=sequence_length,  shuffle_data=True, percentage_val_set=.1)

        dim_vocab = len(vocab)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=hidden_dim)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs,
            training_parameters=training_parameters, batch_callback=_model_batch_callback,
            validation_x_input_int_list_of_sequences=validation_inputs, validation_y_label_int_list_of_sequences=validation_labels)


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
        logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, learning_rate=%f",
            dim_vocab, dim_hidden, sequence_length, learning_rate)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, learning_rate=0.001,
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
        logger.info("Training started. dim_vocab=%d, dim_hidden=%d, sequence_length=%d, learning_rate=%f",
            dim_vocab, dim_hidden, sequence_length, learning_rate)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, learning_rate=0.001,
            batch_size=batch_size, max_epoch=max_epoch, batch_callback=_model_batch_callback)


    def profile_training(self):
        # Creating profile object
        ob = cProfile.Profile()
        ob.enable()

        cProfile.run('profile_training()')

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