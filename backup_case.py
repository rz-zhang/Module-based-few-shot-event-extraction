# Sync TEST
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data

from model import Net

from consts import ARGUMENTS, TRIGGERS
from data_load import ACE2005Dataset, ACE2005DatasetBase, ACE2005DatasetNovel, pad, all_triggers, all_entities, all_postags, tokenizer
from utils import report_to_telegram, build_vocab
from eval import eval, eval_module, eval_gating, eval_backup
from prettytable import PrettyTable


def train(model, iterator, optimizer, criterion, module_mode="module", argu_loss_weight=2):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, srl_arguments, srl_triggers = batch
        optimizer.zero_grad()
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_info, auxiliary_feature, srl_gating_index_li = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                  postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                  triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, srl_triggers=srl_triggers)

        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))
        loss = trigger_loss

        # if len(argument_keys) > 0:
        #     argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
        #     argument_loss = criterion(argument_logits, arguments_y_1d)
        #     input('Look at batch shape')
        #     print(arguments_y_1d.shape)
        #     input('The shape of argument y 1d')
        #     loss = trigger_loss + 2 * argument_loss
        #     if i == 0:
        #         print("=====sanity check for arguments======")
        #         print('arguments_y_1d:', arguments_y_1d)
        #         print("arguments_2d[0]:", arguments_2d[0]['events'])
        #         print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
        #         print("=======================")
        # else:
        #     loss = trigger_loss
        if len(argument_keys) > 0:
            argu_logits_li = []
            for module_arg in ARGUMENTS:
                argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.module_predict_arguments(argument_hidden, argument_keys, arguments_2d, module_arg)
                # argument_loss = criterion(argument_logits, arguments_y_1d)
                # loss += 2 * argument_loss

                # meta classifier
                # module_decisions_logit, module_decisions_y, argument_hat_2d = model.module.meta_classifier(argument_keys, arguments_2d, trigger_info, argument_logits, argument_hat_1d, auxiliary_feature, module_arg)
                # module_decisions_logit = module_decisions_logit.view(-1)
              
                # decision_loss = decision_criterion(module_decisions_logit, module_decisions_y)
                # loss += 2 * decision_loss
                # argument_loss = criterion(argument_logits, arguments_y_1d)
                # loss += argu_loss_weight * argument_loss

                # for gating
                argu_logits_li.append(argument_logits)
            # get the slice of srl triggers
            srl_gating_triggers_li = []
            for srl_idx in srl_gating_index_li:
                srl_gating_triggers_li.append(srl_triggers[srl_idx])
            gating_logits, gating_arguments_y_1d, module_decisions_hat_1d, argument_hat_2d = model.module.module_gating(argu_logits_li, srl_gating_triggers_li, argument_keys, arguments_2d)
            # print('gating logtis',gating_logits)
            # print('gating y 1d', gating_arguments_y_1d)
            gating_loss = criterion(gating_logits, gating_arguments_y_1d)
            loss += argu_loss_weight * gating_loss

            # for module_arg in ARGUMENTS:
            #     argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.module_predict_arguments(argument_hidden, argument_keys, arguments_2d, module_arg)
            #     argument_loss = criterion(argument_logits, arguments_y_1d)
            #     loss += 2 * argument_loss

            # if i == 0:
            #     print("=====sanity check for arguments======")
            #     print('arguments_y_1d:', arguments_y_1d)
            #     print("arguments_2d[0]:", arguments_2d[0]['events'])
            #     print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
            #     print("=======================")
          #else:
              #loss = trigger_loss


        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()

        # if i == 0:
        #     print("=====sanity check======")
        #     print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
        #     print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
        #     print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
        #     print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
        #     print("triggers_2d[0]:", triggers_2d[0])
        #     print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print("seqlens_1d[0]:", seqlens_1d[0])
        #     print("arguments_2d[0]:", arguments_2d[0])
        #     print("=======================")

        if i % 100 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--ft_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train_srl.json")
    parser.add_argument("--devset", type=str, default="data/dev_srl.json")
    parser.add_argument("--testset", type=str, default="data/test_srl.json")
    parser.add_argument("--module_arg", type=str, default="all")
    parser.add_argument("--module_output", type=str, default="module_dir")
    parser.add_argument("--model_load_name", type=str, default="pretrain_attack_5way_model.pt")

    parser.add_argument("--model_save_name", type=str, default="finetune_attack_5way_model.pt")
    parser.add_argument("--result_output", type=str, default="backup_smog_5way_result.txt")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--loss_weight", type=int, default=2)
    parser.add_argument("--module_mode", type=str, default="module")
    parser.add_argument("--eval_mode", type=str, default='backup')

    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")

    parser.add_argument("--mix_train_dev", type=bool, default=True)
    parser.add_argument("--shuffle_dataset", type=bool, default=True)
    parser.add_argument("--dev_split", type=float, default=0.05)
    parser.add_argument("--novel_shot", type=int, default=5)
    #parser.add_argument("--novel_event", type=list, default= ['Justice:Sentence', 'Personnel:Elect', 'Life:Marry', 'Business:Start-Org', 'Personnel:Start-Position', 'Conflict:Demonstrate', 'Justice:Arrest-Jail',  'Justice:Release-Parole', 'Justice:Trial-Hearing', 'Life:Injure'])
    # for backup exp
    parser.add_argument("--novel_event", type=list, default= ['Justice:Convict', 'Conflict:Attack', 'Life:Marry', 'Business:Start-Org', 'Personnel:Start-Position', 'Conflict:Demonstrate', 'Justice:Arrest-Jail',  'Justice:Release-Parole', 'Justice:Trial-Hearing', 'Life:Injure'])
    
    parser.add_argument("--novel_way_num", type=int, default=5)
    

    hp = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hp.module_arg == 'all':
        all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
    elif hp.module_arg in ARGUMENTS:
        all_arguments, argument2idx, idx2argument = build_vocab([hp.module_arg], BIO_tagging=False)

    novel_event=hp.novel_event[:hp.novel_way_num]
    base_event = [item for item in TRIGGERS if item not in novel_event]
   
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)


    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    if not os.path.exists(hp.module_output):
        os.makedirs(hp.module_output)


    dev_f1_max = 0


    metric_output = os.path.join(hp.module_output, hp.result_output)
    model_save_path = os.path.join(hp.module_output, hp.model_load_name)
    model = torch.load(model_save_path)
    if device == 'cuda':
        model = model.cuda()

    #model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # test on pretrain dataset
    test_dataset = ACE2005Dataset(hp.testset, all_arguments, argument2idx)
    #test_dataset = ACE2005DatasetBase(hp.testset, all_arguments, argument2idx, base_event=base_event)
    test_iter = data.DataLoader(dataset=test_dataset,
                            batch_size=hp.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=pad)
    metric, argument_f1, num_proposed, num_correct, num_gold, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2 = eval_backup(model, test_iter, '__', idx2argument)
    print('[Pretrain Test Overall Stats]\n')
    print('F={:.3f}\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(argument_f1, num_proposed, num_correct, num_gold))
    print('[Pretrain Test Attacker CASE Stats]\n')
    print('F={:.3f}\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold))  
    print('[Pretrain Test Target CASE Stats]\n')
    print('F={:.3f}\t Num_prop={}\t Num_correct={}\t Num_gold={}'.format(bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2))  
    


    with open(metric_output, 'a') as fout:
        fout.write("----------------End of Pre-training  TEST on base set------------------\n\n\n")
        fout.write("----------------Finetune on novel set------------------\n")
    # data split
    full_dataset = ACE2005DatasetBase(hp.trainset, all_arguments, argument2idx, hp.devset, base_event=base_event)
    # Creating data indices for training and dev splits:
    dev_split = 0.05
    shuffle_dataset = True
    random_seed= np.random.randint(1,1000)
    fullset_size = len(full_dataset)
    indices = list(range(fullset_size))
    split = int(np.floor(dev_split * fullset_size))
    if hp.shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, dev_indices = indices[split:], indices[:split]
    train_dataset = ACE2005DatasetBase(hp.trainset, all_arguments, argument2idx, hp.devset, train_indices, base_event=base_event)
    dev_dataset = ACE2005DatasetBase(hp.trainset, all_arguments, argument2idx, hp.devset, dev_indices, base_event=base_event)
    # finetune on novel set




    ft_train_dataset = ACE2005DatasetNovel(hp.trainset, all_arguments, argument2idx, novel_event=novel_event, novel_shot=hp.novel_shot)
    ft_dev_dataset = ACE2005DatasetNovel(hp.devset, all_arguments, argument2idx, novel_event=novel_event, novel_shot=5000)
    ft_test_dataset = ACE2005DatasetNovel(hp.testset, all_arguments, argument2idx, novel_event=novel_event, novel_shot=5000)
    samples_weight = ft_train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    ft_train_iter = data.DataLoader(dataset=ft_train_dataset,
                                 batch_size=hp.ft_batch_size,
                                 shuffle=False,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)
    ft_dev_iter = data.DataLoader(dataset=ft_dev_dataset,
                               batch_size=hp.ft_batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    ft_test_iter = data.DataLoader(dataset=ft_test_dataset,
                                batch_size=hp.ft_batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)



    for epoch in range(1, hp.n_epochs + 1):
        dev_table = PrettyTable(['Argument',  'F1', 'Num_proposed', 'Num_correct', 'Num_gold'])
        test_table = PrettyTable(['Argument',  'F1', 'Num_proposed', 'Num_correct', 'Num_gold'])

        train(model, ft_train_iter, optimizer, criterion)

        fname = os.path.join(hp.logdir, str(epoch))
        if hp.eval_mode=='gating':
            print(f"=========GATING eval dev at epoch={epoch}=========")
            metric_dev, dev_f1 = eval_gating(model, ft_dev_iter, fname + '_dev', idx2argument)

            print(f"=========GATING eval test at epoch={epoch}=========")
            metric_test, test_arg_f1 = eval_gating(model, ft_test_iter, fname + '_test', idx2argument)
            if dev_f1 >= dev_f1_max:
                dev_f1_max = dev_f1
                metric_output = os.path.join(hp.module_output, hp.result_output)
                model_save_path = os.path.join(hp.module_output, hp.model_save_name)
                torch.save(model, model_save_path)
                with open(metric_output, 'a') as fout:
                  fout.write(f"=========eval dev at epoch={epoch}=========\n")
                  fout.write(metric_dev)
                  fout.write(f"\n=========eval test at epoch={epoch}=========\n")
                  fout.write(metric_test)
                  fout.write('\n\n')
        elif hp.eval_mode=='module':
            dev_proposed, dev_correct, dev_gold = 0, 0, 0
            test_proposed, test_correct, test_gold = 0, 0, 0
            print(f"=========eval dev at epoch={epoch}=========")
            for module in ARGUMENTS:
              print('---------Argument={}---------'.format(module))
              metric_dev, dev_arg_f1, num_proposed, num_correct, num_gold = eval_module(model, ft_dev_iter, fname + '_dev', module, idx2argument)
              dev_proposed += num_proposed
              dev_correct += num_correct 
              dev_gold += num_gold
              dev_table.add_row([module, round(dev_arg_f1,3), num_proposed, num_correct, num_gold])
            if dev_correct==0 or dev_proposed==0:
              dev_p = 0
            else:
              dev_p = dev_correct/dev_proposed
            dev_r = dev_correct/dev_gold
            if dev_p + dev_r ==0:
              dev_f1 = 0
            else:
              dev_f1 = dev_p*dev_r*2/(dev_p+dev_r)
            dev_table.add_row(['All', round(dev_f1, 3), dev_proposed, dev_correct, dev_gold])
            print(dev_table)
            

            print(f"=========eval test at epoch={epoch}=========")
            for module in ARGUMENTS:
              print('---------Argument={}---------'.format(module))
              metric_test, test_arg_f1, num_proposed, num_correct, num_gold = eval_module(model, ft_test_iter, fname + '_test', module, idx2argument)
              test_proposed += num_proposed
              test_correct += num_correct 
              test_gold += num_gold
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


            if dev_f1 >= dev_f1_max:
              dev_f1_max = dev_f1
              metric_output = os.path.join(hp.module_output, hp.result_output)
              model_save_path = os.path.join(hp.module_output, hp.model_save_name)
              torch.save(model, model_save_path)
              with open(metric_output, 'a') as fout:
                fout.write(f"=========eval dev at epoch={epoch}=========\n")
                fout.write(dev_table.get_string())
                fout.write(f"\n=========eval test at epoch={epoch}=========\n")
                fout.write(test_table.get_string())
                fout.write('\n\n')
        elif hp.eval_mode=='backup':
            print(f"=========BACKUP eval dev at epoch={epoch}=========")
            metric_dev, dev_f1, num_proposed, num_correct, num_gold, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2 = eval_backup(model, test_iter, '__', idx2argument)
            print(f"=========BACKUP eval test at epoch={epoch}=========")
            metric_test, test_f1, num_proposed, num_correct, num_gold, bp_argument_f1, bp_num_proposed, bp_num_correct, bp_num_gold, bp_argument_f1_2, bp_num_proposed_2, bp_num_correct_2, bp_num_gold_2 = eval_backup(model, test_iter, '__', idx2argument)
            
            if dev_f1 >= dev_f1_max:
                dev_f1_max = dev_f1
                metric_output = os.path.join(hp.module_output, hp.result_output)
                model_save_path = os.path.join(hp.module_output, 'finetune_'+hp.model_save_name)
                #torch.save(model, model_save_path)
                with open(metric_output, 'a') as fout:
                  fout.write(f"=========eval dev at epoch={epoch}=========\n")
                  fout.write(metric_dev)
                  fout.write(f"\n=========eval test at epoch={epoch}=========\n")
                  fout.write(metric_test)
                  fout.write('\n\n')

        else:
            print(f"=========MULTI eval dev at epoch={epoch}=========")
            metric_dev, dev_f1 = eval(model, ft_dev_iter, fname + '_dev', idx2argument)

            print(f"=========MULTI eval test at epoch={epoch}=========")
            metric_test, test_arg_f1 = eval(model, ft_test_iter, fname + '_test', idx2argument)
            if dev_f1 >= dev_f1_max:
                dev_f1_max = dev_f1
                metric_output = os.path.join(hp.module_output, hp.result_output)
                model_save_path = os.path.join(hp.module_output, 'finetune_'+hp.model_save_name)
                torch.save(model, model_save_path)
                with open(metric_output, 'a') as fout:
                  fout.write(f"=========eval dev at epoch={epoch}=========\n")
                  fout.write(metric_dev)
                  fout.write(f"\n=========eval test at epoch={epoch}=========\n")
                  fout.write(metric_test)
                  fout.write('\n\n')

        if hp.telegram_bot_token:
            report_to_telegram('[epoch {}] dev\n{}'.format(epoch, metric_dev), TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            report_to_telegram('[epoch {}] test\n{}'.format(epoch, metric_test), TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)


