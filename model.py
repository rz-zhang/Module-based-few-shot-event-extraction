import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_load import idx2trigger, trigger2idx_noBIO, all_srl
from consts import NONE, ARGUMENTS
from utils import find_triggers


class Net(nn.Module):
    def __init__(self, argument2idx=None, idx2argument=None, trigger_size=None, trigger_size_noBIO=None, entity_size=None, srl_size=None, all_postags=None, postag_embedding_dim=50, trigger_embedding_dim=50, srl_embedding_dim=50,  argument_size=None, entity_embedding_dim=50, module_argument_size=3, device=torch.device("cpu")):
        super().__init__()
        self.argument2idx = argument2idx
        self.idx2argument = idx2argument
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=postag_embedding_dim)
        self.trigger_embed = nn.Embedding(num_embeddings=trigger_size_noBIO, embedding_dim=trigger_embedding_dim)
        self.srl_embed = nn.Embedding(num_embeddings=srl_size, embedding_dim=srl_embedding_dim)
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768 + entity_embedding_dim, hidden_size=768 // 2, batch_first=True)

        hidden_size = 768 + entity_embedding_dim + postag_embedding_dim + srl_embedding_dim
        #hidden_size = 768 + entity_embedding_dim + postag_embedding_dim

        argument_hidden_size = hidden_size*2 + trigger_embedding_dim
        #argument_hidden_size = hidden_size*2 

        #hidden_size = 768
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size , trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(argument_hidden_size, argument_size),
            #nn.ReLU(),
            #nn.Linear(argument_size, argument_size),
        )
<<<<<<< HEAD
        self.module_list = nn.ModuleList(
            nn.Sequential(nn.Linear(argument_hidden_size, module_argument_size),) for _ in range(28)
        )
=======

        self.module_list = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),) for _ in range(28)
        )

>>>>>>> 3c95077f331d015dc00ce76a31e91e943407db44
        self.top_select = nn.Sequential(
                nn.Linear(1+1+3+768, 1),
                #nn.Sigmoid()
        )
        self.top_gating = nn.Sequential(nn.Linear(3*len(ARGUMENTS)+len(all_srl)+trigger_embedding_dim, argument_size), )

        self.device = device

    def module_select(self, module_argument):
        if module_argument <= 1:
            return None
        return self.module_list[module_argument-2]

<<<<<<< HEAD
    def srl_embed_2d(self, srl_triggers):
        srl_x_2d = []
        for seq in srl_triggers:
            seq_li = []
            for item in seq:
                item = torch.LongTensor(item).to(self.device)
                item_emb = self.srl_embed(item)
                seq_li.append(item_emb)
            seq_emb = torch.stack(seq_li).mean(dim=0)
            srl_x_2d.append(seq_emb)
        return torch.stack(srl_x_2d, dim=0)
        

    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, srl_triggers, train_mode=True):
=======
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
>>>>>>> 3c95077f331d015dc00ce76a31e91e943407db44
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        postags_x_2d = self.postag_embed(postags_x_2d)
        entity_x_2d = self.entity_embed(entities_x_3d)
        
        srl_x_2d = self.srl_embed_2d(srl_triggers)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(tokens_x_2d)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]

        x = torch.cat([enc, entity_x_2d, postags_x_2d, srl_x_2d], 2)
        #x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        # x = enc
        # logits = self.fc2(x + enc)
        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        trigger_info = [] # for meta classifier
        auxiliary_feature = [] # for meta classifier
        srl_gating_li = []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            # if train_mode:
            #     predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in triggers_y_2d[i].tolist()]) # real triggers
            # else:
            #     predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()]) # predicted triggers
            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()]) # predicted triggers


            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str, original_trigger = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                # trigger embedding
                trigger_tensor = self.trigger_embed(torch.LongTensor([trigger2idx_noBIO[t_type_str]]).to(self.device))

                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]
                    argument_hidden.append(torch.cat([event_tensor, entity_tensor, trigger_tensor.mean(dim=0)]))
                    #argument_hidden.append(torch.cat([event_tensor, entity_tensor]))

                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))
                    trigger_info.append(original_trigger)
                    auxiliary_feature.append(enc[i, 0: ,].mean(dim=0))
                    srl_gating_li.append(i)

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, srl_gating_li

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = self.argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)
        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == self.argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d


    def module_predict_arguments(self, argument_hidden, argument_keys, arguments_2d, module):
        argument_hidden = torch.stack(argument_hidden)
        module_fc_argument = self.module_select(self.argument2idx[module])
        argument_logits = module_fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = 1 # for the single argument, 1 is the value of 'O' class
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                      if self.idx2argument[a_type_idx] == module:
                        a_label = 2 # for the single argument, the argument2idx dict is {'Argument': 2, 'O': 1, '[PAD]': 0}
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label != 2: # for the single argument, 1 is the value of 'O' class
              continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def srl_gating_onehot(self, srl_triggers):
        # Input: srl_triggers (num_candidates * max_len_in_batch)
        srl_onthot_2d = []
        # for seq_tags in srl_triggers:
        #     seq_onehot = [0] * len(all_srl)
        #     for tag in seq_tags:
        #         seq_onehot[tag] = 1
        #     srl_onthot_2d.append(seq_onehot)
        for sequence in srl_triggers:
            seq_onehot = [0] * len(all_srl)
            for verb_tags in sequence:
                for tag in verb_tags:
                    seq_onehot[tag] = 1
            srl_onthot_2d.append(seq_onehot)
        return srl_onthot_2d

    def module_gating(self, module_logits, srl_triggers, argument_keys, arguments_2d):
        # Input: 
        #     module_logits (num_candidates, 3)
        #     srl_triggers (num_candidates, max_len_in_batch)
        module_logits = torch.cat(module_logits, dim=1) # shape (num_candidate, num_module*3)
        srl_onehot_2d = self.srl_gating_onehot(srl_triggers) # shape (num_candidates, num_all_srl)
        srl_onehot_2d = torch.LongTensor(srl_onehot_2d).to(self.device)
        # aux trigger
        aux_trigger = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            trigger_tensor = self.trigger_embed(torch.LongTensor([trigger2idx_noBIO[t_type_str]]).to(self.device))
            aux_trigger.append(trigger_tensor.mean(dim=0))
        aux_trigger = torch.stack(aux_trigger)

        #gating_input = torch.cat([module_logits, srl_onehot_2d], dim=1) #shape (num_candidates, num_module*3 + num_all_srl)
        gating_input = torch.cat([module_logits, srl_onehot_2d, aux_trigger], dim=1)
        gating_logits = self.top_gating(gating_input) 
        module_decisions_hat_1d = gating_logits.argmax(-1)
        #module_decisions_hat = torch.round(torch.sigmoid(gating_logits)) #(num_candidates, num_module)
        # update arguments_hat_2d
        gating_arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = self.argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            gating_arguments_y_1d.append(a_label)
        gating_arguments_y_1d = torch.LongTensor(gating_arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, module_decisions_hat_1d.cpu().numpy()):
            if a_label == self.argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return gating_logits, gating_arguments_y_1d, module_decisions_hat_1d, argument_hat_2d



    def meta_classifier(self, argument_keys, arguments_2d, trigger_info, argument_logits, argument_hat_1d, auxiliary_feature, module):
        # Input: trigger_info, argument_logits, auxiliary features
        # Output: decision to each module
        trigger_info = torch.FloatTensor(trigger_info).to(self.device)
        trigger_info = trigger_info.view(-1, 1)
        module_info = self.argument2idx[module]
        module_info = torch.full(trigger_info.shape, module_info, dtype=torch.float).to(self.device)
        auxiliary_feature = torch.stack(auxiliary_feature)
        
        meta_input = torch.cat([trigger_info, module_info, argument_logits, auxiliary_feature], dim = 1)
        #meta_input = torch.cat([trigger_info, auxiliary_feature, module_info], dim = 1)
        #meta_input = torch.cat([trigger_info, module_info], dim = 1)

        module_decisions_logit = self.top_select(meta_input)
        module_decisions_hat = torch.round(torch.sigmoid(module_decisions_logit))
        module_decisions_y = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            decision = 0
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        decision = 1
                        break
            module_decisions_y.append(decision)
        module_decisions_y = torch.FloatTensor(module_decisions_y).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label, decision in zip(argument_keys, argument_hat_1d.cpu().numpy(), module_decisions_hat.detach().cpu().numpy()):
            if decision == 0:
                continue
            if a_label!=2:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))
        

        return module_decisions_logit, module_decisions_y, argument_hat_2d



# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
