import numpy as np
import torch
from torch.utils import data
import json
import copy

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS, SRL_ARGUMENTS
from utils import build_vocab, srl_find_trigger, srl_find_argument, srl_split_role
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_srl, srl2idx, idx2srl = build_vocab(SRL_ARGUMENTS, BIO_tagging=False)
all_triggers_noBIO, trigger2idx_noBIO, idx2trigger_noBIO = build_vocab(TRIGGERS, BIO_tagging=False)
#all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))



class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath, all_arguments, argument2idx, fpath_mix=None, indices=None):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.srl_arguments_li, self.srl_triggers_li = [], [], [], [], [], [], []
        if fpath_mix:
            fpath_list = [fpath, fpath_mix]
        else:
            fpath_list = [fpath]
      
        for _fpath in fpath_list:
            with open(_fpath, 'r') as f:
                data = json.load(f)
                for item in data:
                    words = item['words']
                    entities = [[NONE] for _ in range(len(words))]
                    triggers = [NONE] * len(words)
                    postags = item['pos-tags']
                    arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_triggers_single_seq = [] # ex. [[verb1_srl_tag1, ..., verb1_srl_tagn],[verb2_srl_tag1, ..., verb2_srl_tagn],...]

                    for entity_mention in item['golden-entity-mentions']:
                        arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                        for i in range(entity_mention['start'], entity_mention['end']):
                            entity_type = entity_mention['entity-type']
                            if i == entity_mention['start']:
                                entity_type = 'B-{}'.format(entity_type)
                            else:
                                entity_type = 'I-{}'.format(entity_type)

                            if len(entities[i]) == 1 and entities[i][0] == NONE:
                                entities[i][0] = entity_type
                            else:
                                entities[i].append(entity_type)

                    for event_mention in item['golden-event-mentions']:
                        for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                            trigger_type = event_mention['event_type']
                            if i == event_mention['trigger']['start']:
                                triggers[i] = 'B-{}'.format(trigger_type)
                            else:
                                triggers[i] = 'I-{}'.format(trigger_type)

                        event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                        arguments['events'][event_key] = []
                        for argument in event_mention['arguments']:
                            role = argument['role']
                            if role.startswith('Time'):
                                role = role.split('-')[0]
                            if role in all_arguments:
                              arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                    for srl_item in item['srl']:
                        srl_trigger_start, srl_trigger_end = srl_find_trigger(srl_item['tags'])
                        if srl_trigger_start != None:
                            event_key = (srl_trigger_start, srl_trigger_end, srl_item['verb'])
                            argument_list = srl_find_argument(srl_item['tags'])
                            srl_arguments['events'][event_key] = []
                            for argument in argument_list:
                                srl_arguments['events'][event_key].append(argument)

                    for srl_item in item['srl']:
                        srl_triggers = []
                        for tag in srl_item['tags']:
                            # split the BI tag and the CR middle tag
                            tag = srl_split_role(tag)
                            srl_triggers.append(tag)
                        srl_triggers_single_seq.append(srl_triggers)
                    if srl_triggers_single_seq==[]:
                        srl_triggers_single_seq=[[]] # reshape 


                    self.sent_li.append([CLS] + words + [SEP])
                    self.entities_li.append([[PAD]] + entities + [[PAD]])
                    self.postags_li.append([PAD] + postags + [PAD])
                    self.triggers_li.append(triggers)
                    self.arguments_li.append(arguments)
                    self.srl_arguments_li.append(srl_arguments)
                    self.srl_triggers_li.append(srl_triggers_single_seq)

        if indices:
            new_sent_li = [self.sent_li[index] for index in indices]
            new_entities_li = [self.entities_li[index] for index in indices]
            new_postags_li = [self.postags_li[index] for index in indices]
            new_triggers_li = [self.triggers_li[index] for index in indices]
            new_arguments_li = [self.arguments_li[index] for index in indices]
            new_srl_arguments_li = [self.srl_arguments_li[index] for index in indices]
            new_srl_triggers_li = [self.srl_triggers_li[index] for index in indices]
            self.sent_li = new_sent_li
            self.entities_li = new_entities_li
            self.postags_li = new_postags_li
            self.triggers_li = new_triggers_li
            self.arguments_li = new_arguments_li
            self.srl_arguments_li = new_srl_arguments_li
            self.srl_triggers_li = new_srl_triggers_li



    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, srl_arguments, srl_triggers_single_seq = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.srl_arguments_li[idx], self.srl_triggers_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]

        # convert srl tag to index
        new_srl = copy.deepcopy(srl_triggers_single_seq)
        for i, item in enumerate(srl_triggers_single_seq):
            for j, tag in enumerate(item):
                if tag!='V':
                    new_srl[i][j] = srl2idx[tag]
                else:
                    new_srl[i][j] = srl2idx[NONE]
                # new_srl[i][j] = srl2idx[tag]


        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, srl_arguments, new_srl

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


class ACE2005DatasetBase(data.Dataset):
    def __init__(self, fpath, all_arguments, argument2idx, fpath_mix=None, indices=None, base_event=None):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.srl_arguments_li, self.srl_triggers_li = [], [], [], [], [], [], []
        if fpath_mix:
            fpath_list = [fpath, fpath_mix]
        else:
            fpath_list = [fpath]
      
        for _fpath in fpath_list:
            with open(_fpath, 'r') as f:
                data = json.load(f)
                for item in data:
                    words = item['words']
                    entities = [[NONE] for _ in range(len(words))]
                    triggers = [NONE] * len(words)
                    postags = item['pos-tags']
                    arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_triggers_single_seq = [] # ex. [[verb1_srl_tag1, ..., verb1_srl_tagn],[verb2_srl_tag1, ..., verb2_srl_tagn],...]
                    base_flag = True

                    for entity_mention in item['golden-entity-mentions']:
                        arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                        for i in range(entity_mention['start'], entity_mention['end']):
                            entity_type = entity_mention['entity-type']
                            if i == entity_mention['start']:
                                entity_type = 'B-{}'.format(entity_type)
                            else:
                                entity_type = 'I-{}'.format(entity_type)

                            if len(entities[i]) == 1 and entities[i][0] == NONE:
                                entities[i][0] = entity_type
                            else:
                                entities[i].append(entity_type)

                    for event_mention in item['golden-event-mentions']:
                        for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                            trigger_type = event_mention['event_type']
                            if trigger_type not in base_event:
                                base_flag = False    

                            if i == event_mention['trigger']['start']:
                                triggers[i] = 'B-{}'.format(trigger_type)
                            else:
                                triggers[i] = 'I-{}'.format(trigger_type)

                        event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])

                        arguments['events'][event_key] = []
                        for argument in event_mention['arguments']:
                            role = argument['role']
                            if role.startswith('Time'):
                                role = role.split('-')[0]
                            if role in all_arguments:
                              arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                    for srl_item in item['srl']:
                        srl_trigger_start, srl_trigger_end = srl_find_trigger(srl_item['tags'])
                        if srl_trigger_start != None:
                            event_key = (srl_trigger_start, srl_trigger_end, srl_item['verb'])
                            argument_list = srl_find_argument(srl_item['tags'])
                            srl_arguments['events'][event_key] = []
                            for argument in argument_list:
                              srl_arguments['events'][event_key].append(argument)

                    for srl_item in item['srl']:
                        srl_triggers = []
                        for tag in srl_item['tags']:
                            # split the BI tag and the CR middle tag
                            tag = srl_split_role(tag)
                            srl_triggers.append(tag)
                        srl_triggers_single_seq.append(srl_triggers)
                    if srl_triggers_single_seq==[]:
                        srl_triggers_single_seq=[[]] # reshape 

                    if base_flag:
                        self.sent_li.append([CLS] + words + [SEP])
                        self.entities_li.append([[PAD]] + entities + [[PAD]])
                        self.postags_li.append([PAD] + postags + [PAD])
                        self.triggers_li.append(triggers)
                        self.arguments_li.append(arguments)
                        self.srl_arguments_li.append(srl_arguments)
                        self.srl_triggers_li.append(srl_triggers_single_seq)

        if indices:
            new_sent_li = [self.sent_li[index] for index in indices]
            new_entities_li = [self.entities_li[index] for index in indices]
            new_postags_li = [self.postags_li[index] for index in indices]
            new_triggers_li = [self.triggers_li[index] for index in indices]
            new_arguments_li = [self.arguments_li[index] for index in indices]
            new_srl_arguments_li = [self.srl_arguments_li[index] for index in indices]
            new_srl_triggers_li = [self.srl_triggers_li[index] for index in indices]
            self.sent_li = new_sent_li
            self.entities_li = new_entities_li
            self.postags_li = new_postags_li
            self.triggers_li = new_triggers_li
            self.arguments_li = new_arguments_li
            self.srl_arguments_li = new_srl_arguments_li
            self.srl_triggers_li = new_srl_triggers_li



    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, srl_arguments, srl_triggers_single_seq = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.srl_arguments_li[idx], self.srl_triggers_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        # convert srl tag to index
        new_srl = copy.deepcopy(srl_triggers_single_seq)
        for i, item in enumerate(srl_triggers_single_seq):
            for j, tag in enumerate(item):
                new_srl[i][j] = srl2idx[tag]
                # if tag!='V':
                #     new_srl[i][j] = srl2idx[tag]
                # else:
                #     new_srl[i][j] = srl2idx[NONE]

        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, srl_arguments, new_srl

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


class ACE2005DatasetNovel(data.Dataset):
    def __init__(self, fpath, all_arguments, argument2idx, fpath_mix=None, indices=None, novel_event=None, novel_shot=5):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.srl_arguments_li, self.srl_triggers_li = [], [], [], [], [], [], []
        if fpath_mix:
            fpath_list = [fpath, fpath_mix]
        else:
            fpath_list = [fpath]
      
        novel_event_count = {item: 0 for item in novel_event}
        for _fpath in fpath_list:
            with open(_fpath, 'r') as f:
                data = json.load(f)
                for item in data:
                    words = item['words']
                    entities = [[NONE] for _ in range(len(words))]
                    triggers = [NONE] * len(words)
                    postags = item['pos-tags']
                    arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }
                    srl_triggers_single_seq = [] # ex. [[verb1_srl_tag1, ..., verb1_srl_tagn],[verb2_srl_tag1, ..., verb2_srl_tagn],...]
                    novel_flag = False
                    temp_novel_event = [] # count the novel event in this data item  

                    for entity_mention in item['golden-entity-mentions']:
                        arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                        for i in range(entity_mention['start'], entity_mention['end']):
                            entity_type = entity_mention['entity-type']
                            if i == entity_mention['start']:
                                entity_type = 'B-{}'.format(entity_type)
                            else:
                                entity_type = 'I-{}'.format(entity_type)

                            if len(entities[i]) == 1 and entities[i][0] == NONE:
                                entities[i][0] = entity_type
                            else:
                                entities[i].append(entity_type)

                    for event_mention in item['golden-event-mentions']:
                        for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                            trigger_type = event_mention['event_type']
                            if i == event_mention['trigger']['start']:
                                triggers[i] = 'B-{}'.format(trigger_type)
                            else:
                                triggers[i] = 'I-{}'.format(trigger_type)
                        if trigger_type in novel_event:
                            novel_flag = True
                            temp_novel_event.append(trigger_type)
                            #novel_event_count[trigger_type] += 1

                        event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])

                        arguments['events'][event_key] = []
                        for argument in event_mention['arguments']:
                            role = argument['role']
                            if role.startswith('Time'):
                                role = role.split('-')[0]
                            if role in all_arguments:
                              arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                    for srl_item in item['srl']:
                        srl_trigger_start, srl_trigger_end = srl_find_trigger(srl_item['tags'])
                        if srl_trigger_start != None:
                            event_key = (srl_trigger_start, srl_trigger_end, srl_item['verb'])
                            argument_list = srl_find_argument(srl_item['tags'])
                            srl_arguments['events'][event_key] = []
                            for argument in argument_list:
                                srl_arguments['events'][event_key].append(argument)


                    for srl_item in item['srl']:
                        srl_triggers = []
                        for tag in srl_item['tags']:
                            # split the BI tag and the CR middle tag
                            tag = srl_split_role(tag)
                            srl_triggers.append(tag)
                        srl_triggers_single_seq.append(srl_triggers)
                    if srl_triggers_single_seq==[]:
                        srl_triggers_single_seq=[[]] # reshape 

                    if novel_flag:
                        for key, value in novel_event_count.items():
                            if int(value) < novel_shot and key in set(temp_novel_event):
                                self.sent_li.append([CLS] + words + [SEP])
                                self.entities_li.append([[PAD]] + entities + [[PAD]])
                                self.postags_li.append([PAD] + postags + [PAD])
                                self.triggers_li.append(triggers)
                                self.arguments_li.append(arguments)
                                self.srl_arguments_li.append(srl_arguments)
                                self.srl_triggers_li.append(srl_triggers_single_seq)
                                for temp_novel_event_item in set(temp_novel_event):
                                      novel_event_count[temp_novel_event_item] += 1
                      # if max(novel_event_count.values()) > novel_shot:
                      #     print('-------Novel Event Type Overload-------\n')
                      #     print(novel_event_count)

        if indices:
            new_sent_li = [self.sent_li[index] for index in indices]
            new_entities_li = [self.entities_li[index] for index in indices]
            new_postags_li = [self.postags_li[index] for index in indices]
            new_triggers_li = [self.triggers_li[index] for index in indices]
            new_arguments_li = [self.arguments_li[index] for index in indices]
            new_srl_arguments_li = [self.srl_arguments_li[index] for index in indices]
            new_srl_triggers_li = [self.srl_triggers_li[index] for index in indices]
            self.sent_li = new_sent_li
            self.entities_li = new_entities_li
            self.postags_li = new_postags_li
            self.triggers_li = new_triggers_li
            self.arguments_li = new_arguments_li
            self.srl_arguments_li = new_srl_arguments_li
            self.srl_triggers_li = new_srl_triggers_li

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, srl_arguments, srl_triggers_single_seq = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.srl_arguments_li[idx], self.srl_triggers_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        # convert srl tag to index
        new_srl = copy.deepcopy(srl_triggers_single_seq)
        for i, item in enumerate(srl_triggers_single_seq):
            for j, tag in enumerate(item):
                # if tag!='V':
                #     new_srl[i][j] = srl2idx[tag]
                # else:
                #     new_srl[i][j] = srl2idx[NONE]
                new_srl[i][j] = srl2idx[tag]

        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, srl_arguments, new_srl

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)

def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arg_2d, srl_triggers_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]
        for j, item in enumerate(srl_triggers_2d[i]):
            srl_triggers_2d[i][j] = item + [srl2idx[PAD]] * (maxlen - len(item))

    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d, srl_arg_2d, srl_triggers_2d
