import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_load import idx2trigger
from consts import NONE
from utils import find_triggers


class Net(nn.Module):
    def __init__(self, argument2idx=None, idx2argument=None, trigger_size=None, entity_size=None, all_postags=None, postag_embedding_dim=50, argument_size=None, entity_embedding_dim=50, module_argument_size=3, device=torch.device("cpu")):
        super().__init__()
        self.argument2idx = argument2idx
        self.idx2argument = idx2argument
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=postag_embedding_dim)
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768 + entity_embedding_dim, hidden_size=768 // 2, batch_first=True)

        hidden_size = 768 + entity_embedding_dim + postag_embedding_dim
        #hidden_size = 768
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
            #nn.ReLU(),
            #nn.Linear(argument_size, argument_size),
        )
        self.fc_argument_0 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_1 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_2 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_3 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_4 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_5 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_6 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_7 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_8 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_9 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_10 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_11 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_12 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_13 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_14 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_15 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_16 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_17 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_18 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_19 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_20 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_21 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_22 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_23 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_24 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_25 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_26 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)
        self.fc_argument_27 = nn.Sequential(nn.Linear(hidden_size * 2, module_argument_size),)

        self.device = device

    def module_select(self, module_argument):
        module_list = [None, None, self.fc_argument_0, self.fc_argument_1, self.fc_argument_2, self.fc_argument_3,
              self.fc_argument_4, self.fc_argument_5, self.fc_argument_6, self.fc_argument_7,
              self.fc_argument_8, self.fc_argument_9, self.fc_argument_10, self.fc_argument_11,
              self.fc_argument_12, self.fc_argument_13, self.fc_argument_14, self.fc_argument_15,
              self.fc_argument_16, self.fc_argument_17, self.fc_argument_18, self.fc_argument_19,
              self.fc_argument_20, self.fc_argument_21, self.fc_argument_22, self.fc_argument_23,
              self.fc_argument_24, self.fc_argument_25, self.fc_argument_26, self.fc_argument_27,
              ]
        return module_list[module_argument]

    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        postags_x_2d = self.postag_embed(postags_x_2d)
        entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(tokens_x_2d)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]

        x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        # x = enc
        # logits = self.fc2(x + enc)

        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

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
