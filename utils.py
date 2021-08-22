import pandas as pd
import numpy as np
import os, time, random
import datetime
from contextlib import contextmanager
from tqdm import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import torch.cuda.amp as amp

from loss import *
from scheduler import *


id_name = 'ID' # column name in dataframe
label_name = ['label'] # some task may have mutiple labels
sequence_name = "sequence"
gpus = list(range(torch.cuda.device_count()))
print("Available GPUs", gpus)


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None


def Metric(preds, labels):
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)

    AUC = roc_auc_score(labels,preds)
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    AUPRC = auc(recalls, precisions)
    return AUC, AUPRC


def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None


class TaskDataset:
    def __init__(self, df, protein_data, label_name):
        self.df = df
        self.protein_data = protein_data
        self.label_name = label_name

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        pdb_id = self.df.loc[idx,'ID']
        protein_node_features,protein_edge_features,protein_dist_matrix,protein_masks,labels = self.protein_data[pdb_id]

        return {
            'PDB_ID': pdb_id,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_EDGE_FEAT': protein_edge_features,
            'PROTEIN_DIST': protein_dist_matrix,
            'PROTEIN_MASK': protein_masks,
            'LABEL': labels,
        }

    def collate_fn(self, batch):
        """
        自定义数据合并方法，将数据集中的数据通过Padding构造成相同Size
        """
        pdb_ids = []
        protein_node_features = torch.stack([item['PROTEIN_NODE_FEAT'] for item in batch], dim=0)
        protein_edge_features = torch.stack([item['PROTEIN_EDGE_FEAT'] for item in batch], dim=0)
        protein_dist_matrix = torch.stack([item['PROTEIN_DIST'] for item in batch], dim=0)
        protein_masks = torch.stack([item['PROTEIN_MASK'] for item in batch], dim=0)
        labels = torch.stack([item['LABEL'] for item in batch], dim=0)

        for item in batch:
            pdb_ids.append(item['PDB_ID'])

        return pdb_ids, protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks, labels


# main function
def NN_train_and_predict(train, test, protein_data, model_class, config, logit=False, output_root='./output/', run_id=None, args=None):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if not run_id:
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    hidden_unit = config['hidden_unit']
    fc_layer = config['fc_layer']
    self_atten_layer = config['self_atten_layer']
    attention_heads = config['attention_heads']
    num_neighbor = config['num_neighbor']
    fc_dropout = config['fc_dropout']
    attention_dropout = config['attention_dropout']
    node_dim = config['node_dim']

    id_name = config['id_name']
    obj_max = config['obj_max']
    epochs = config['epochs']
    smoothing = config['smoothing']
    patience = config['patience']
    lr = config['lr']
    batch_size = config['batch_size']
    folds = config['folds']
    seed = config['seed']

    if train is not None:
        os.system(f'cp ./*.py {output_path}')
        os.system(f'cp ./*.sh {output_path}')

        oof = train[[id_name, sequence_name]]
        oof['fold'] = -1
        if isinstance(label_name, list):
            for l in label_name:
                oof[l] = 0.0
                oof[l] = oof[l].astype(np.float32)
        else:
            oof[label_name] = 0.0
            oof[label_name] = oof[label_name].astype(np.float32)
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w', buffering=1)
        log.write(str(config) + '\n')

        all_valid_metric = []

        kf = KFold(n_splits = folds, shuffle=True, random_state=seed)

        train_folds = []
        for fold, (train_index, val_index) in enumerate(kf.split(train, train[label_name])):
            print("\n========== Fold " + str(fold + 1) + " ==========")

            train_dataset = TaskDataset(train.loc[train_index].reset_index(drop=True), protein_data, label_name)
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=5000)   # when set to sampler, shuffle is False.
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, sampler=sampler, shuffle=False, drop_last=True, num_workers=args.num_workers, prefetch_factor=2)

            valid_dataset = TaskDataset(train.loc[val_index].reset_index(drop=True), protein_data, label_name)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
            
            if model_class.__name__ in ['GTM']:
                model = model_class(node_dim, hidden_unit, len(label_name), fc_layer, self_atten_layer, attention_heads, num_neighbor, fc_dropout, attention_dropout)

            scheduler = Adam12()

            model.cuda()
            # optimizer = scheduler.schedule(model, 0, epochs)[0]

            if args.use_apm:
                scaler = amp.GradScaler()
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr, weight_decay=0.00001, eps=1e-5)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            # loss_tr = nn.BCEWithLogitsLoss()
            # loss_tr = FocalLoss()
            # loss_tr = FocalSymmetricLovaszHardLogLoss()
            # loss_feature = PairLoss()
            loss_tr = nn.BCEWithLogitsLoss(reduction='none')
            if obj_max == 1:
                best_valid_metric = 0
            else:
                best_valid_metric = 1e9
            not_improve_epochs = 0
            if args.train:
                for epoch in range(epochs):
                    train_loss = 0.0
                    train_num = 0
                    model.train()
                    scheduler.step(model,epoch,epochs)

                    train_pdb_ids = []
                    train_preds = []
                    train_Y = []
                    bar = tqdm(train_dataloader)
                    for i, data in enumerate(bar):
                        optimizer.zero_grad()
                        protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks, y = [d.cuda() for d in data[1:]]
                        if smoothing > 0:
                            y = (1-smoothing)*y + smoothing/max(len(label_name),2)

                        if args.use_apm:
                            with amp.autocast():
                                outputs = model(protein_node_features,protein_edge_features,protein_dist_matrix,protein_masks)
                                loss = loss_tr(outputs,y)*protein_masks
                                loss = loss.sum() / protein_masks.sum()

                            if str(loss.item()) == 'nan': continue
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks)
                            loss = loss_tr(outputs, y) * protein_masks
                            loss = loss.sum() / protein_masks.sum()
                            loss.backward()
                            optimizer.step()

                        if logit:
                            outputs = outputs.sigmoid() # outputs.shape = (batch_size, max_len)

                        train_pdb_ids.extend(data[0])

                        train_seq_preds = torch.masked_select(outputs, protein_masks.bool()) # masked_select 会将bsize个样本的预测值展平，直接batch内所有样本合并成一个list了
                        train_preds.append(train_seq_preds.detach().cpu().numpy())

                        train_seq_y = torch.masked_select(y, protein_masks.bool())
                        train_Y.append(train_seq_y.clone().detach().cpu().numpy())

                        train_num += len(train_seq_y)
                        train_loss += len(train_seq_y) * loss.item()

                        bar.set_description('loss: %.4f' % (loss.item()))

                    train_loss /= train_num
                    train_preds = np.concatenate(train_preds)
                    train_preds = train_preds
                    train_Y = np.concatenate(train_Y)
                    train_Y = train_Y

                    train_metric = Metric(train_preds, train_Y) # 一个epoch其实等于我之前的好多个epoch


                    # eval
                    model.eval()
                    valid_preds = []
                    valid_Y = []
                    for data in tqdm(valid_dataloader):
                        protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks, y = [d.cuda() for d in data[1:]]
                        with torch.no_grad():
                            if logit:
                                outputs = model(protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks).sigmoid()
                            else:
                                outputs = model(protein_node_features,protein_edge_features,protein_dist_matrix,protein_masks)

                        valid_seq_y = torch.masked_select(y, protein_masks.bool())
                        valid_Y.append(valid_seq_y.detach().cpu().numpy())

                        valid_seq_preds = torch.masked_select(outputs, protein_masks.bool())
                        valid_preds.append(valid_seq_preds.detach().cpu().numpy())

                    valid_preds = np.concatenate(valid_preds)
                    valid_Y = np.concatenate(valid_Y)
                    valid_metric = Metric(valid_preds, valid_Y)

                    if obj_max * (valid_metric[1]) > obj_max * best_valid_metric: # use AUPRC
                        if len(gpus) > 1:
                            torch.save(model.module.state_dict(), output_path + 'fold%s.ckpt'%fold)
                        else:
                            torch.save(model.state_dict(), output_path + 'fold%s.ckpt'%fold)
                        not_improve_epochs = 0
                        best_valid_metric = valid_metric[1]
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f, valid_auprc:%.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,train_metric[0],train_metric[1],valid_metric[0],valid_metric[1]))
                    else:
                        not_improve_epochs += 1
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f, valid_auprc:%.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,train_metric[0],train_metric[1],valid_metric[0],valid_metric[1],not_improve_epochs))
                        if not_improve_epochs >= patience:
                            break


            # 用最好的epoch再测试一下validation，并存下一些结果
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda' if torch.cuda.is_available() else 'cpu') )

            if model_class.__name__ in ['GTM']:
                model = model_class(node_dim, hidden_unit, len(label_name), fc_layer, self_atten_layer, attention_heads, num_neighbor, fc_dropout, attention_dropout)

            model.cuda()
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()

            valid_preds = []
            valid_outputs = []
            valid_Y = []
            for data in tqdm(valid_dataloader):
                protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks, y = [d.cuda() for d in data[1:]]
                with torch.no_grad():
                    if logit:
                        outputs = model(protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks).sigmoid()
                    else:
                        outputs = model(protein_node_features,protein_edge_features,protein_dist_matrix,protein_masks)

                valid_outputs.append(outputs.detach().cpu().numpy()) # outputs = bz * max_len

                valid_seq_y = torch.masked_select(y, protein_masks.bool())
                valid_Y.append(valid_seq_y.detach().cpu().numpy())

                valid_seq_preds = torch.masked_select(outputs, protein_masks.bool())
                valid_preds.append(valid_seq_preds.detach().cpu().numpy())

            valid_outputs = np.concatenate(valid_outputs) # shape = (num_samples * max_len)
            valid_preds = np.concatenate(valid_preds)
            valid_Y = np.concatenate(valid_Y)
            valid_mean = np.mean(valid_preds)
            valid_metric = Metric(valid_preds, valid_Y)
            Write_log(log,'[fold %s] best_valid_auc: %.6f, best_valid_auprc: %.6f, best_valid_mean: %.6f'%(fold, valid_metric[0], valid_metric[1], valid_mean))

            all_valid_metric.append(valid_metric[0]) # AUC
            oof.loc[val_index,label_name] = [valid_outputs[i, :len(oof.loc[val_idx,sequence_name])].tolist() for i,val_idx in enumerate(val_index)]
            oof.loc[val_index,'fold'] = fold
            train_folds.append(fold)


        mean_valid_metric = np.mean(all_valid_metric)
        Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))
        oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv',index=False)

        if test is None:
            log.close()
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'folds':folds,'metric':[mean_valid_metric],'lb':[np.nan],'remark':[config['remark']]})
        if not os.path.exists(output_root + 'experiment_log.csv'):
            log_df.to_csv(output_root + 'experiment_log.csv', index=False)
        else:
            log_df.to_csv(output_root + 'experiment_log.csv',index=False, mode='a', header=None)

    if test is not None:
        if train is None:
            log = open(output_path + 'test.log','w', buffering=1)
            Write_log(log,str(config)+'\n')
        sub = test[[id_name, sequence_name]]
        if isinstance(label_name,list):
            for l in label_name:
                sub[l] = 0.0
        else:
            sub[label_name] = 0.0

        test_dataset = TaskDataset(test, protein_data, label_name)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
        models = []
        for fold in range(folds):
            if not os.path.exists(output_path + 'fold%s.ckpt'%fold):
                continue

            if model_class.__name__ in ['GTM']:
                model = model_class(node_dim, hidden_unit, len(label_name), fc_layer, self_atten_layer, attention_heads, num_neighbor, fc_dropout, attention_dropout)

            model.cuda()
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda') )
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()
            models.append(model)
        print('model count:',len(models))

        test_preds = []
        test_outputs = []
        test_Y = []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks, y = [d.cuda() for d in data[1:]]

                if logit:
                    outputs = [model(protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks).sigmoid() for model in models]
                else:
                    outputs = [model(protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks) for model in models]

                outputs = torch.stack(outputs,0).mean(0) # 5个模型预测结果求平均,最终shape=(bsize, max_len)
                test_outputs.append(outputs.detach().cpu().numpy())
            
                test_seq_preds = torch.masked_select(outputs, protein_masks.bool())
                test_preds.append(test_seq_preds.cpu().detach().numpy())

                test_seq_y = torch.masked_select(y, protein_masks.bool())
                test_Y.append(test_seq_y.cpu().detach().numpy())

        test_outputs = np.concatenate(test_outputs) # shape = (num_samples * max_len)
        test_preds = np.concatenate(test_preds)
        test_Y = np.concatenate(test_Y)
        sub[label_name] = [test_outputs[i, :len(sub.loc[i,sequence_name])].tolist() for i in range(len(sub))]

        if label_name[0] in test.columns:
            test_metric = Metric(test_preds, test_Y)
            Write_log(log,'test_auc:%.6f, test_auprc:%.6f'%(test_metric[0],test_metric[1]))

        sub.to_csv(output_path+'submission.csv',index=False)
        log.close()
    else:
        sub = None

    return oof,sub
