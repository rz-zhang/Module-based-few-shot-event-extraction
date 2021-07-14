import os
import argparse
import torch
import torch.nn as nn
import copy
import random

from torch.utils import data
from model import Net
from consts import ARGUMENTS, TRIGGERS, EVENT_SCHEMA
from data_load import ACE2005Dataset, ACE2005DatasetBase, ACE2005DatasetNovel, pad, all_triggers, all_entities, all_postags, idx2trigger
from utils import calc_metric, find_triggers, build_vocab
from prettytable import PrettyTable


def find_overlap(arguments_pred_in):
    "e.g. argument_pred_in = (i, t_type_str, index, a_type_idx)"
    arguments_pred = copy.deepcopy(arguments_pred_in)
    overlap_part_index, no_overlap_part_index = [], []
    argu_dict = {}
    total_len = len(arguments_pred)
    for i in range(total_len):
        key = (arguments_pred[i][0], arguments_pred[i][2])
        if key in argu_dict.keys():
            argu_dict[key].append(i)
        else:
            argu_dict[key] = [i]
        # if key in argu_dict.keys():
        #     argu_dict[key] += 1
        # else:
        #     argu_dict[key] = 1

    for key, value in argu_dict.items():
        if len(value)>1:
            temp_overlap_part_index = []
            for item in value:
                temp_overlap_part_index.append(item)
            overlap_part_index.append(temp_overlap_part_index)
        else:
            no_overlap_part_index.append(value[0])
    # for key, value in argu_dict.items():
    #     if value > 1:
    #         for i, x in enumerate(arguments_pred):
    #             if key == (x[0], x[2]):
    #               overlap_part_index.append(i)
    #overlap_part = [arguments_pred[i] for i in range(total_len) if i in overlap_part_index]
    #no_overlap_part = [arguments_pred[i] for i in range(total_len) if i in no_overlap_part_index]

    return overlap_part_index, no_overlap_part_index, argu_dict

def eval_token_level(arguments_true, arguments_pred):
    """arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))"""
    new_argu_true = []
    new_argu_pred = []
    for item in arguments_true:
        i, t_type_str, a_start, a_end, a_type_idx = item
        for index in range(a_start+1, a_end+1):
            new_argu_true.append((i, t_type_str, index, a_type_idx))
    for item in arguments_pred:
        i, t_type_str, a_start, a_end, a_type_idx = item
        for index in range(a_start+1, a_end+1):
            new_argu_pred.append((i, t_type_str, index, a_type_idx))

    overlap_part_index, no_overlap_part_index, argu_dict = find_overlap(new_argu_true)
    print('Length of the overlapping:\n {}'.format(len(overlap_part_index)))
    print('Length of the NON-overlapping:\n {}'.format(len(no_overlap_part_index)))

    # No_overlap_part
    total_len = len(new_argu_true)
    no_overlap_part = [new_argu_true[i] for i in range(total_len) if i in no_overlap_part_index]
    p, r, f = calc_metric(new_argu_true, no_overlap_part)
    print('Precison = {}\n Recall = {}\n F1 = {}\n'.format(p, r, f))

    # Overlap_part Random select
    overlap_part_select = []
    mismatch_count = 0
    mismatch_set = []
    for items in overlap_part_index:
        temp_a_type = new_argu_true[items[0]][-1]
        mismatch = []
        mismatch.append(idx2argument[temp_a_type])
        for item in items:
            if temp_a_type != new_argu_true[item][-1]:
                # print('*** Mismatch ***')
                # print('*** Current A_TYPE = {}***'.format(temp_a_type))
                # print(new_argu_true[item])
                mismatch.append(idx2argument[new_argu_true[item][-1]])
                mismatch_count += 1
            overlap_part_select.append(new_argu_true[item])
        mismatch = set(mismatch)
        mismatch_set.append(mismatch)

            
    print('MISMATCH COUNT = {}'.format(mismatch_count))
    print('MISMATCH_SET = {}'.format(mismatch_set))
        
    # for items in overlap_part_index:
    #     temp_len = len(items)
    #     select_index = items[random.randint(0, temp_len-1)]
    #     try:
    #         overlap_part_select.append(new_argu_true[select_index])
    #     except:
    #         print(select_index)
    #         print(total_len)

    p, r, f = calc_metric(new_argu_true, overlap_part_select)
    print('Precison = {}\n Recall = {}\n F1 = {}\n'.format(p, r, f))

    # Select max probablity

def eval_token_level_all(arguments_true, arguments_pred):
    """arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))"""
    new_argu_true = []
    new_argu_pred = []
    for item in arguments_true:
        i, t_type_str, a_start, a_end, a_type_idx = item
        for index in range(a_start+1, a_end+1):
            new_argu_true.append((i, t_type_str, index, a_type_idx))
    for item in arguments_pred:
        i, t_type_str, a_start, a_end, a_type_idx = item
        for index in range(a_start+1, a_end+1):
            new_argu_pred.append((i, t_type_str, index, a_type_idx))
    p, r, f = calc_metric(new_argu_true, new_argu_pred)
    print('Precison = {}\n Recall = {}\n F1 = {}\n'.format(p, r, f))


def eval(model, iterator, fname, idx2argument):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arguments, srl_triggers = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, ___ = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                          postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                          triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, srl_triggers=srl_triggers,train_mode=False)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    #if idx2argument[a_type_idx] in EVENT_SCHEMA[t_type_str]:
                    arguments_pred.append((i, t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write('#arguments#{}\n'.format(arguments['events']))
            fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))
    # print('ARGUMENTS TRUE', arguments_true)
    # print('ARGUMENT_PRED', arguments_pred)

    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    print('[argument identification]')
    arguments_true = [(item[0], item[1], item[2], item[3]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2], item[3]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    # final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    # with open(final, 'w') as fout:
    #     result = open("temp", "r").read()
    #     fout.write("{}\n".format(result))
    #     fout.write(metric)
    # os.remove("temp")
    return metric, argument_f1


def eval_gating(model, iterator, fname, idx2argument):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    out_schema_count = 0
    out_schema_dict = {item: [] for item in enumerate(TRIGGERS)}
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arguments, srl_triggers = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, srl_gating_index_li = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                          postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                          triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, srl_triggers=srl_triggers,train_mode=False)
            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argu_logits_li = []
                for module_arg in ARGUMENTS:
                    argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.module_predict_arguments(argument_hidden, argument_keys, arguments_2d, module_arg)
                    # argument_loss = criterion(argument_logits, arguments_y_1d)
                    # loss += argu_loss_weight * argument_loss

                    # for gating
                    argu_logits_li.append(argument_logits)
                # get the slice of srl triggers
                srl_gating_triggers_li = []
                for srl_idx in srl_gating_index_li:
                    srl_gating_triggers_li.append(srl_triggers[srl_idx])
                gating_logits, gating_arguments_y_1d, module_decisions_hat_1d, argument_hat_2d = model.module.module_gating(argu_logits_li, srl_gating_triggers_li, argument_keys, arguments_2d)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    if idx2argument[a_type_idx] in EVENT_SCHEMA[t_type_str]:
                        arguments_pred.append((i, t_type_str, a_start, a_end, a_type_idx))
                    # if idx2argument[a_type_idx] not in EVENT_SCHEMA[t_type_str]:
                    #     out_schema_dict[t_type_str].append(idx2argument[a_type_idx])

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write('#arguments#{}\n'.format(arguments['events']))
            fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))
    # print('ARGUMENTS TRUE', arguments_true)
    # print('ARGUMENT_PRED', arguments_pred)
    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    print('[argument identification]')
    arguments_true = [(item[0], item[1], item[2], item[3]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2], item[3]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    # final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    # with open(final, 'w') as fout:
    #     result = open("temp", "r").read()
    #     fout.write("{}\n".format(result))
    #     fout.write(metric)
    # os.remove("temp")
    return metric, argument_f1

def eval_backup(model, iterator, fname, idx2argument):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arguments, srl_triggers = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, srl_gating_index_li = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                          postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                          triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, srl_triggers=srl_triggers,train_mode=False)
            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argu_logits_li = []
                for module_arg in ARGUMENTS:
                    argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.module_predict_arguments(argument_hidden, argument_keys, arguments_2d, module_arg)
                    # argument_loss = criterion(argument_logits, arguments_y_1d)
                    # loss += argu_loss_weight * argument_loss

                    # for gating
                    argu_logits_li.append(argument_logits)
                # get the slice of srl triggers
                srl_gating_triggers_li = []
                for srl_idx in srl_gating_index_li:
                    srl_gating_triggers_li.append(srl_triggers[srl_idx])
                gating_logits, gating_arguments_y_1d, module_decisions_hat_1d, argument_hat_2d = model.module.module_gating(argu_logits_li, srl_gating_triggers_li, argument_keys, arguments_2d)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    #if idx2argument[a_type_idx] in EVENT_SCHEMA[t_type_str]:
                    arguments_pred.append((i, t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write('#arguments#{}\n'.format(arguments['events']))
            fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))
    backup_arguments_true = []
    backup_arguments_pred = []
    for item in arguments_true:
        if idx2argument[item[-1]] == 'Attacker':
            backup_arguments_true.append(item)
    for item in arguments_pred:
        if idx2argument[item[-1]] == 'Attacker':
            backup_arguments_pred.append(item)
    print('[BACKUP ARGUMENTS CLASSIFICATION]')
    bp_argument_p, bp_argument_r, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold = calc_metric(backup_arguments_true, backup_arguments_pred, num_flag=True)
    print('Attacker F={:.3f}\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold)) 

    backup_arguments_true_2 = []
    backup_arguments_pred_2 = []
    for item in arguments_true:
        if idx2argument[item[-1]] == 'Target':
            backup_arguments_true_2.append(item)
    for item in arguments_pred:
        if idx2argument[item[-1]] == 'Target':
            backup_arguments_pred_2.append(item)
    bp_argument_p_2, bp_argument_r_2, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2 = calc_metric(backup_arguments_true_2, backup_arguments_pred_2, num_flag=True)
    print('F={:.3f}\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2)) 

    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1, num_proposed, num_correct, num_gold = calc_metric(arguments_true, arguments_pred, num_flag=True)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    print('[argument identification]')
    arguments_true = [(item[0], item[1], item[2], item[3]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2], item[3]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[Attacker classification] \tP={:.3f}\tR={:.3f}\tF1={:.3f}\n[backup argument numbers]\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_p, bp_argument_r, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold)
    metric += '[Target classification] \tP={:.3f}\tR={:.3f}\tF1={:.3f}\n[backup argument numbers]\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_p_2, bp_argument_r_2, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2)

    # final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    # with open(final, 'w') as fout:
    #     result = open("temp", "r").read()
    #     fout.write("{}\n".format(result))
    #     fout.write(metric)
    # os.remove("temp")
    return metric, argument_f1, num_proposed, num_correct, num_gold, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2


def eval_module(model, iterator, fname, module, idx2argument):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arguments, srl_triggers = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, ___ = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                  postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                  triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, srl_triggers=srl_triggers,train_mode=False)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.module_predict_arguments(argument_hidden, argument_keys, arguments_2d, module)
                #module_decisions_logit, module_decisions_y, argument_hat_2d = model.module.meta_classifier(argument_keys, arguments_2d, trigger_info, argument_logits, argument_hat_1d, auxiliary_feature, module)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    # strict metric
                    #arguments_true.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))
                    # relaxed metric
                    if idx2argument[a_type_idx] == module:
                        arguments_true.append((i, t_type_str, a_start, a_end, 2))
                    #else:
                    #  arguments_true.append((i, t_type_str, a_start, a_end, 1))


            #print(arguments_hat)
            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    # stric metric
                    # arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))
                    # relaxed metric
                    #if idx2argument[a_type_idx] == module:
                    #if idx2argument[a_type_idx] in EVENT_SCHEMA[t_type_str]:
                    arguments_pred.append((i, t_type_str, a_start, a_end, a_type_idx)) # 2 is the specific argument idx in module network
                    # else:
                    #   print(idx2argument[a_type_idx])
                    #   arguments_pred.append((i, t_type_str, a_start, a_end, 1)) 

            # if len(arguments_pred) == 0:
            #   print('---batch {} -----'.format(i))
            #   print(arguments_hat)

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write('#arguments#{}\n'.format(arguments['events']))
            fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1, num_proposed, num_correct, num_gold = calc_metric(arguments_true, arguments_pred, num_flag=True)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    
    #print('[trigger identification]')
    # triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    # triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    # trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    #print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    #print('[argument identification]')
    # strcit metric
    #arguments_true = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_true]
    #arguments_pred = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_pred]
    # relax metric
    # arguments_true = [(item[0], item[1], item[2], item[3]) for item in arguments_true]
    # arguments_pred = [(item[0], item[1], item[2], item[3]) for item in arguments_pred]
    # argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    #print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    # metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    # metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    # final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    # with open(final, 'w') as fout:
    #     result = open("temp", "r").read()
    #     fout.write("{}\n".format(result))
    #     fout.write(metric)
    # os.remove("temp")
    return metric, argument_f1, num_proposed, num_correct, num_gold #,arguments_true, arguments_pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--testset", type=str, default="data/test_srl.json")
    parser.add_argument("--model_path", type=str, default="latest_model.pt")
    parser.add_argument("--module_arg", type=str, default="all")
    parser.add_argument("--eval_mode", type=str, default='finetune')
    parser.add_argument("--eval_base", type=str, default='novel')
    parser.add_argument("--novel_event", type=list, default= ['Justice:Convict', 'Personnel:Elect', 'Life:Marry', 'Business:Start-Org', 'Personnel:Start-Position'])
    #parser.add_argument("--novel_event", type=list, default= ['Justice:Sentence', 'Personnel:Elect', 'Life:Marry', 'Business:Start-Org', 'Personnel:Start-Position'])



    hp = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(hp.model_path):
        print('Warning: There is no model on the path:', hp.model_path, 'Please check the model_path parameter')
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    model = torch.load(hp.model_path)

    if device == 'cuda':
        model = model.cuda()

    if hp.module_arg == 'all':
        all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
    elif hp.module_arg in ARGUMENTS:
        all_arguments, argument2idx, idx2argument = build_vocab([hp.module_arg], BIO_tagging=False)

    if hp.eval_mode=='finetune':
        novel_event = hp.novel_event
        if hp.eval_base =='novel':
            test_dataset = ACE2005DatasetNovel(hp.testset, all_arguments, argument2idx, novel_event=novel_event, novel_shot=1000)
        elif hp.eval_base =='all':
            test_dataset = ACE2005Dataset(hp.testset, all_arguments, argument2idx)
        else:
            base_event = [item for item in TRIGGERS if item not in novel_event]
            test_dataset = ACE2005DatasetBase(hp.testset, all_arguments, argument2idx, base_event=base_event)
    else:
        test_dataset = ACE2005Dataset(hp.testset, all_arguments, argument2idx)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    if hp.eval_mode=='finetune':
        print(f"=========eval test=========")
        # eval(model, test_iter, 'eval_test')
        test_table = PrettyTable(['Argument',  'F1', 'Num_proposed', 'Num_correct', 'Num_gold'])
        test_proposed, test_correct, test_gold = 0, 0, 0
        arguments_true_all, arguments_pred_all = [], []
        for module in ARGUMENTS:
            print('---------Argument={}---------'.format(module))
            metric_test, test_arg_f1, num_proposed, num_correct, num_gold = eval_module(model, test_iter, '_', module, idx2argument)
            test_proposed += num_proposed
            test_correct += num_correct 
            test_gold += num_gold
            #arguments_true_all.extend(arguments_true)
            #arguments_pred_all.extend(arguments_pred)
            #eval_token_level_all(arguments_true, arguments_pred)
            test_table.add_row([module, round(test_arg_f1,3), num_proposed, num_correct, num_gold])
        if test_correct==0 or test_proposed==0:
            test_p = 0
        else:
            test_p = test_correct/test_proposed
        test_r = test_correct/test_gold
        if test_p + test_r ==0:
            test_f1 =0
        else:
            test_f1 = test_p*test_r*2/(test_p+test_r)
        test_table.add_row(['All', round(test_f1, 3), test_proposed, test_correct, test_gold])
        print(test_table)
    else:
        print(f"=========MULTI eval test =========")
        fname = os.path.join(hp.logdir, '_test')
        metric_test, test_arg_f1 = eval(model, test_iter, fname, idx2argument)
    #print('Token Level Evaluation\n')
    #eval_token_level_all(arguments_true_all, arguments_pred_all)
 
