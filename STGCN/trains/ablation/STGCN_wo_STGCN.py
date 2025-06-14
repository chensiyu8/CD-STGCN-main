import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class STGCN_wo_STGCN():
    def __init__(self, args):

        self.args = args
        self.args.tasks = 'M'
        self.criterionMSE = nn.MSELoss()
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]

        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n ]

        optimizer = optim.Adam(
            [{"params": bert_params_decay, "weight_decay": self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
             {"params": bert_params_no_decay, "weight_decay": 0.0, 'lr': self.args.learning_rate_bert},
             {'params': model_params_other, "weight_decay": self.args.weight_decay_other, "lr": self.args.learning_rate_other}]
           )
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)
                    # compute loss
                    loss = 0.0

                    # multi-task loss
                    #M-loss
                    loss += eval('self.args.M') * self.criterion(outputs['M'], labels['M'])
                    #print(loss)
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
            train_loss = train_loss / len(dataloader['train'])

            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} "
            )
            #print(train_loss)
            if train_loss != train_loss:
                logger.info('loss=nan,skip!!!')
                break
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' % (m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())

                    loss = 0.0
                    # multi-task loss
                    for m in 'M':
                        loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode + "-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' % (m) + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.args.tasks[0]]
        eval_results['Loss'] = round(eval_loss, 4)

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results


