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


test_text = '''
黄蓉不禁神往，轻轻叹了口气，问道：“师父，您在船上与西毒比武，干吗不用出来？”洪七公道：“使这棒法是我帮的大事，况且即使不用，西毒也未必胜得了我。谁料到他如此卑鄙无耻，我两次救他性命，他反在背后伤我。”黄蓉见师父神色黯然，要让他分心，便道：“师父，您将棒法教会蓉儿，我去杀了西毒，给您报仇。”
洪七公淡淡一笑，捡起地下一根枯柴，身子斜倚石壁，口中传诀，手上比划，将三十六路棒法一路路地都教了她。他知黄蓉聪敏异常，又怕自己命不久长，是以一口气地传授完毕。那打狗棒法名字虽然陋俗，但变化精微，招术奇妙，实是古往今来武学中的第一等功夫，若非如此，焉能作为丐帮帮主历代相传的镇帮之宝？黄蓉虽绝顶聪明，也只记得个大要，其中玄奥之处，一时之间却哪能领会得了？
等到传毕，洪七公叹了一口气，汗水涔涔而下，说道：“我教得太过简略，到底不好，可是……可是也只能这样了。”斜身倒地，晕了过去。黄蓉大惊，连叫：“师父，师父！”抢上去扶时，只觉他手足冰冷，脸无血色，气若游丝。黄蓉在数日之间迭遭变故，伏在师父胸口竟哭不出来，耳听得他一颗心还在微微跳动，忙伸掌在他胸口用力一揿一放，以助呼吸，就在这紧急关头，忽听得身后有声轻响，一只手伸过来拿她手腕。她全神贯注地相救师父，欧阳克何时进来，竟全不知晓，这时她忘了身后站着的是一头豺狼，却回头道：“师父不成啦，快想法子救他。”
欧阳克见她回眸求恳，一双大眼中含着眼泪，神情楚楚可怜，心中不由得一荡，俯身看洪七公时，见他脸如白纸，两眼上翻，心下更喜。他与黄蓉相距不到半尺，只感到她吹气如兰，闻到的尽是她肌肤上的香气，几缕柔发在她脸上掠过，心中痒痒的再也忍耐不住，伸左臂就去搂她纤腰。
黄蓉一惊，沉肘反掌，用力拍出，乘他转头闪避，已自跃起。欧阳克原本忌惮洪七公了得，不敢对黄蓉用强，这时见他神危力竭，十成中倒已死了九成半，再无顾忌，晃身拦在洞口，笑道：“好妹子，我对你本来决不想动蛮，但你如此美貌，我实在熬不得了，你让我亲一亲。”说着张开左臂，一步步地逼来。
黄蓉吓得心中怦怦乱跳，寻思：“今日之险，又远过赵王府之时，看来只有自求了断，只是不手刃此獠，总不甘心。”翻手入怀，将郭靖那柄短剑与一把镀金钢针都拿在手里。欧阳克脸露微笑，脱下长衣当做兵器，又逼近了两步。黄蓉站着不动，待他又跨出一步，足底尚未着地之际，身子倏地向左横闪。欧阳克跟着过来，黄蓉左手空扬，见他挥起长衣抵挡钢针，身子已如箭离弦，急向洞外奔去。
'''

dim_hidden = 64
fixed_learning_rate = 0.0001

def save_model(model):
    filepath_template = './saved_numpy_models/model_%s.pkl'
    filepath = filepath_template % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def model_training_batch_callback(model, prompt, char_to_id_map, id_to_char_map, output_length=100):
    generated_text = ModelUtils.generate_text(model, prompt, char_to_id_map, id_to_char_map, output_length=100)


def test_simple_training():
    vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs = ModelUtils.prepare_data_from_text(test_text, 50)
    dim_vocab = len(vocab)

    def _model_batch_callback(model):
        model_training_batch_callback(model, '洪七公脸如白纸', char_to_id_map, id_to_char_map, output_length=60)

    rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=dim_hidden)

    logger.debug('input_ids_seqs=%s\ntype=%s', input_id_seqs, type(input_id_seqs))

    rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs, fixed_learning_rate=fixed_learning_rate,
        batch_size=1, batch_callback=_model_batch_callback)


class TestNumpyRnnTextGeneration(unittest.TestCase):

    def test_simple_training(self):
        test_simple_training()

    def profile_simple_training(self):
        # Creating profile object
        ob = cProfile.Profile()
        ob.enable()

        cProfile.run('test_simple_training()')

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