import numpy as np
import copy
from consts import NONE, PAD, TRIGGERS


def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def calc_metric(y_true, y_pred, num_flag=False):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    #print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    if num_flag:
      return precision, recall, f1, num_proposed, num_correct, num_gold
    else:
      return precision, recall, f1


# def find_triggers(labels):
#     """
#     :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
#     :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
#     """
#     all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
#     result = []
#     original_labels = copy.deepcopy(labels)
#     labels = [label.split('-') for label in labels]

#     for i in range(len(labels)):
#         if labels[i][0] == 'B':
#             result.append([i, i + 1, labels[i][1], trigger2idx[original_labels[i]]])

#     for item in result:
#         j = item[1]
#         while j < len(labels):
#             if labels[j][0] == 'I':
#                 j = j + 1
#                 item[1] = j
#             else:
#                 break

#     return [tuple(item) for item in result]

def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
    result = []
    original_labels = copy.deepcopy(labels)

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][2:], trigger2idx[original_labels[i]]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]

def srl_split_role(tag):
    if tag[:2]=='B-' or tag[:2]=='I-':
        tag = tag[2:]
    if tag[:2]=='C-' or tag[:2]=='R-':
        tag=tag[2:]
    return tag

def srl_find_trigger(srl_tags):
      start_flag = True
      tag_start, tag_end = None, None
      for i, tag in enumerate(srl_tags):
        if tag.split('-')[-1]=='V' and start_flag:
          tag_start = i
          start_flag = False
        if not start_flag and tag.split('-')[-1]!='V':
          tag_end = i
          break
      return tag_start, tag_end

def srl_find_argument(srl_tags):
      argument_list = []
      for i, tag in enumerate(srl_tags):
        BIO_prefix = tag.split('-')[0]
        abstract_role = srl_split_role(tag)
        if BIO_prefix == 'B' and abstract_role!='V':
          argument_list.append((i, i+1, abstract_role))
      for i, item in enumerate(argument_list):
        j = item[1]
        while j < len(srl_tags):
          if srl_tags[j].split('-')[0]=='I':
            j = j+1
            argument_list[i] = (item[0], j, item[2])
          else:
            break
      return argument_list

# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)
