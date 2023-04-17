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
郭靖存着忌惮之心，不敢跟着进击，神态恭谨，说道：“请前辈赐教。”
黄蓉存心要扰乱裘千仞心神，叫道：“靖哥哥，别跟这糟老头子客气！”
裘千仞成名以来，谁敢当面呼他“糟老头子”？大怒之下，便要纵身过去发掌相击，但转念想起自己身份，冷笑一声，先出右手虚引，再发左手摩眉掌，见郭靖侧身闪避，引手立时钩拿回撤，摩眉掌顺手搏进，转身坐盘，右手迅即挑出，已变塌掌。
黄蓉叫道：“那有什么稀奇？这是‘通臂六合掌’中的‘孤雁出群’！”裘千仞这掌法正是“通臂六合掌”，乃从“通臂五行掌”中变化出来。招数虽不奇，他却已在这掌法上花了数十载寒暑之功。所谓通臂，乃双臂贯为一劲之意，倒不是真的左臂可缩至右臂，右臂可缩至左臂。郭靖见他右手发出，左手往右手贯劲，左手随发之时，右手往回带撤，以增左手之力，双手确有相互应援、连环不断之巧，一来见过他诸般奇技，二来应敌时识见不足，心下怯了，不敢还手招架，记得洪七公所教的“悔”字诀和“退”字诀，不住倒退相避。
裘千仞心道：“这少年一掌碎椅，原来只是力大，武功平常得紧。”随即“穿掌闪劈”、“撩阴掌”、“跨虎蹬山”，越打越显精神。黄蓉见郭靖要败，心中焦急，走近他身边，只要他一遇险招，立时上前相助。郭靖闪开对方斜身蹬足，见黄蓉脸色有异，大见关切，心神微分，裘千仞得势不容情，一招“白蛇吐信”，啪的一掌，平平正正地击中郭靖胸口。黄蓉和江南六怪、陆氏父子齐声惊呼，心想以他功力之深，这一掌正好击在胸口要害，郭靖不死必伤。
郭靖吃了这掌，也大惊失色，但双臂振处，胸口竟不感如何疼痛，大惑不解。黄蓉见他突然发楞，以为必是让这死老头的掌力震昏了，忙抢上扶住，叫道：“靖哥哥，你怎样？”心中一急，两道泪水流了下来。
郭靖却道：“没事！我再试试。”挺起胸膛，走到裘千仞面前，叫道：“你是铁掌老英雄，再打我一掌。”裘千仞大怒，运劲使力，嘭的一声，又在郭靖胸口狠击一掌。郭靖哈哈大笑，叫道：“师父，蓉儿，这老儿武功稀松平常。他不打我倒也罢了，打我一掌，却漏了底。”一语方毕，左臂横扫，逼到裘千仞的身前，叫道：“你也吃我一掌！”
裘千仞见他左臂扫来，口中却说“吃我一掌”，心道：“你臂中套拳，谁不知道？”双手搂怀，来撞他左臂。哪知郭靖这招“龙战于野”是降龙十八掌中十分奥妙的功夫，左臂右掌，均可实可虚，非拘一格，见敌人挡他左臂，右掌忽起，也是嘭的一声，正击在他右臂连胸之处，裘千仞的身子如纸鹞断线般直向门外飞去。

众人惊叫声中，门口突然出现了一人，伸手抓住裘千仞的衣领，大踏步走进厅来，将他在地下一放，凝然而立，脸上冷冷的全无笑容。
众人瞧这人时，只见她长发披肩，抬头仰天，正是铁尸梅超风。
众人心头凛然，见她身后还跟着一人，那人身材高瘦，身穿青色布袍，脸色古怪之极，两颗眼珠似乎尚能微微转动，除此之外，肌肉口鼻，尽皆僵硬如木石，直是一个死人头装在活人的躯体上，令人一见之下，登时一阵凉气从背脊上直冷下来，人人的目光与这张脸孔相触，便都不敢再看，立时将头转开，心中怦怦乱跳。
陆庄主万料不到裘千仞名满天下，口出大言，竟如此不堪一击，本在又好气又好笑，见梅超风蓦地到来，虽容貌已不大识得，但瞧这模样，料来必定是她，心中惊惧哀伤，一时俱集。完颜康见到师父，心中大喜，上前拜见。众人见他二人竟以师徒相称，均感诧异。陆庄主双手一拱，说道：“梅师姊，十余年前相别，今日终又重会，陈师哥可好？”六怪与郭靖听他叫梅超风为师姊，面面相觑，无不凛然。柯镇恶心道：“今日我们落入了圈套，梅超风一人已不易敌，何况更有她的师弟。”黄蓉却暗暗点头：“这庄主的武功文学、谈吐行事，无一不是学我爹爹，我早就疑心他与我家必有渊源，果然是我爹爹的弟子。”
梅超风冷然道：“说话的可是陆乘风陆师弟？”陆庄主道：“正是兄弟，师姊别来无恙？”梅超风道：“说什么别来无恙？我眼睛瞎了，你瞧不出来吗？你玄风师哥也早给人害死了，这可称了你心意么？”陆乘风又惊又喜，惊的是黑风双煞横行天下，怎会栽在敌人手里？喜的是强敌少了一人，而剩下的也双目已盲，想到昔日桃花岛同门学艺的情形，不禁叹了口气，说道：“害死陈师哥的对头是谁？师姊可报了仇么？”梅超风道：“我正在到处找寻他们。”陆乘风道：“小弟当得相助一臂之力，待报了本门怨仇之后，咱们再来清算你我的旧帐。”梅超风哼了一声。
'''

# Main training parameters
dim_hidden = 128
learning_rate = 0.004
sequence_length = 40
batch_size = 10
max_epoch=20


def model_training_batch_callback(model, prompt, char_to_id_map, id_to_char_map, output_length=100):
    generated_text = ModelUtils.generate_text_beam_search(model, prompt, char_to_id_map, id_to_char_map, output_length=100)
    logger.info("Generated text=\n\n%s\n", generated_text)
    ModelUtils.save_model(model, char_to_id_map, id_to_char_map)



class TestNumpyRnnTextGeneration(unittest.TestCase):

    def test_simple_training(self):
        training_parameters = {
            'thread_worker_count': 2,
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

        text_generation_prompt = '洪七公脸如白纸'
        sequence_length = 12
        hidden_dim = 64

        vocab, char_to_id_map, id_to_char_map, input_id_seqs, label_id_seqs, validation_inputs, validation_labels = ModelUtils.prepare_data_from_text(
            text=test_text, sequence_length=sequence_length, shuffle_data=True, percentage_val_set=0.1)

        dim_vocab = len(vocab)

        def _model_batch_callback(model):
            model_training_batch_callback(model, text_generation_prompt, char_to_id_map, id_to_char_map, output_length=100)

        rnn_model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=hidden_dim)

        rnn_model.train(x_input_int_list_of_sequences=input_id_seqs, y_label_int_list_of_sequences=label_id_seqs,
            training_parameters=training_parameters, batch_callback=_model_batch_callback,
            validation_x_input_int_list_of_sequences=validation_inputs, validation_y_label_int_list_of_sequences=validation_labels)


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