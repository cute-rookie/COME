import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import torch
from functions.losses import SetCriterion

import configs
from functions.hashing import get_hamm_dist, calculate_mAP, calculate_fairness, calculate_recall
from utils import io
from utils.misc import AverageMeter, Timer
from models.alexnet import AdversarialNetwork
from torch import optim


def norm(filt):
    # filt (dim, out_dim)
    filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(1, filt.shape[1])
    return filt / filt_norm


def cal(filt):
    filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(1, filt.shape[1])
    norm_mat = torch.matmul(filt_norm.transpose(1, 0), filt_norm)
    inner_pro = torch.matmul(filt.transpose(1, 0), filt)
    return inner_pro / norm_mat


def loss_mhs_weight_reg(filt):
    # filt (output_dim, input_dim)
    filt = filt.transpose(1, 0)  # (in, out)
    filt = norm(filt)
    inner_pro = cal(filt)
    final = (2.0 - 2.0 * inner_pro)
    final -= torch.triu(final)
    nonzeros = torch.where(final != 0)
    target = torch.min(final[nonzeros])
    mask = final.eq(target)
    loss = -(final * mask.detach()).sum()

    return loss


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 1.0


def grad_reverse(x):
    return GradReverse.apply(x)


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def get_codebook(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    """
    brute force to find centroid with furthest distance
    :param nclass:
    :param nbit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    codebook = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    while i < nclass:
        print(i, end='\r')
        c = torch.randn(nbit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, codebook[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            codebook[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    codebook = codebook[torch.randperm(nclass)]
    return codebook


def calculate_accuracy(logits, hamm_dist, labels, loss_param):
    if loss_param['multiclass']:
        pred = logits.topk(5, 1, True, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        acc = correct[:5].view(-1).float().sum(0, keepdim=True) / logits.size(0)

        pred = hamm_dist.topk(5, 1, False, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        cbacc = correct[:5].view(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
    else:
        acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        cbacc = (hamm_dist.argmin(1) == labels.argmax(1)).float().mean()

    return acc, cbacc


def train_hashing(optimizer, model, optim_sa, pred_net_sa, codebook, train_loader, loss_param, config):
    model.train()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    # criterion = OrthoHashLoss(**loss_param)
    # criterion2 = FairSupConLoss(temperature=0.1).cuda()
    criterion_main = SetCriterion(config=config).cuda()

    pred_net_sa.train()
    pred_net_sa.cuda()

    for i, (data, ta, sa) in enumerate(train_loader):
        print(f'iter: {i}/{len(train_loader)}')
        timer.tick()
        labels = ta
        # clear gradient
        optimizer.zero_grad()
        optim_sa.zero_grad()
        #
        # data[0] = data[0].to(device)
        # data[1] = data[1].to(device)
        # data = torch.cat([data[0], data[1]], dim=0)
        # label1 = torch.cat([labels, labels], dim=0)
        # label1 =label1.to(device)
        # sa1 = torch.cat([sa,sa], dim=0)
        # sa1 = sa1.to(device)

        label1 = labels.to(device)

        data, labels = data.to(device), labels.to(device)
        ta, sa = ta.to(device), sa.to(device)

        logits, codes, lvl_features = model(data)

        # loss1 = criterion(logits, codes, label1)
        loss1 = criterion_main(logits, ta, sa, lvl_features, codes)

        loss = loss1

        loss.backward()
        optimizer.step()
        optim_sa.step()

        hamm_dist = get_hamm_dist(codes, codebook, normalize=True)
        acc, cbacc = calculate_accuracy(logits, hamm_dist, label1, loss_param)

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        # meters['loss_ce'].update(criterion1.losses['ce'].item(), data.size(0))
        # meters['loss_ce'].update(criterion_main.losses['loss_ce'].item(), data.size(0))
        # meters['loss_final_feature_MI'].update(criterion_main.losses['loss_final_feature_MI'].item(), data.size(0))
        # meters['loss_lvl'].update(criterion_main.losses['loss_lvl'].item(), data.size(0))
        # meters['loss_attributes_feature_MI'].update(criterion_main.losses['loss_attributes_feature_MI'].item(),
        #                                             data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Train [{i + 1}/{len(train_loader)}] '
              f'T: {meters["loss_total"].avg:.4f} '
              # f'CE: {meters["loss_ce"].avg:.4f} '
              # f'FMI: {meters["loss_final_feature_MI"].avg:.4f} '
              # f'LVL: {meters["loss_lvl"].avg:.4f} '
              # f'LAT: {meters["loss_attributes_feature_MI"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(CB): {meters["cbacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()

    meters['total_time'].update(total_timer.total)

    return meters


def test_hashing(model, codebook, test_loader, loss_param, db=False, return_codes=False, config=None):
    model.eval()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    ret_codes = []
    ret_labels = []
    ret_sa_labels = []
    ret_sa = []
    ret_ta = []

    # criterion = OrthoHashLoss(**loss_param)
    criterion_main = SetCriterion(config=config).cuda()

    for i, (data, ta, sa) in enumerate(test_loader):
        if i % 20:
            print(f'iter: {i}/{len(test_loader)}')
        timer.tick()
        ta = ta.to(device)
        sa = sa.to(device)
        labels = ta
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)
            logits, codes, lvl_features = model(data)

            bs, nbit = codes.size()

            # loss = criterion(logits, codes, labels)
            loss = criterion_main(logits, ta, sa, lvl_features, codes)

            hamm_dist = get_hamm_dist(codes, codebook, normalize=True)
            acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param)

            if return_codes:
                ret_codes.append(codes)
                ret_labels.append(labels)
                ret_sa_labels.append(sa)
                ret_sa.append(torch.tensor(sa, dtype=torch.long))
                ret_ta.append(torch.tensor(ta, dtype=torch.long))

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        # meters['loss_ce'].update(criterion_main.losses['loss_ce'].item(), data.size(0))
        # meters['loss_final_feature_MI'].update(criterion_main.losses['loss_final_feature_MI'].item(), data.size(0))
        # meters['loss_lvl'].update(criterion_main.losses['loss_lvl'].item(), data.size(0))
        # meters['loss_attributes_feature_MI'].update(criterion_main.losses['loss_attributes_feature_MI'].item(),
        #                                             data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Test [{i + 1}/{len(test_loader)}] '
              f'T: {meters["loss_total"].avg:.4f} '
              # f'CE: {meters["loss_ce"].avg:.4f} '
              # f'FMI: {meters["loss_final_feature_MI"].avg:.4f} '
              # f'LVL: {meters["loss_lvl"].avg:.4f} '
              # f'LAT: {meters["loss_attributes_feature_MI"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(CB): {meters["cbacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    meters['total_time'].update(total_timer.total)

    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels),
            'sa_labels': torch.cat(ret_sa_labels),
        }

        sa = torch.cat(ret_sa, dim=0).detach().cpu().numpy()
        ta = torch.cat(ret_ta, dim=0).detach().cpu().numpy()

        return meters, res

    return meters


def prepare_dataloader(config):
    logging.info('Creating Datasets')
    n_labeled_samples = config['sample']['labeled_sample_num']
    train_dataset = configs.dataset(config, filename='train_' + str(n_labeled_samples) + '.txt', transform_mode='train')

    separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
    config['dataset_kwargs']['separate_multiclass'] = False
    test_dataset = configs.dataset(config, filename='test_' + str(n_labeled_samples) + '.txt', transform_mode='test')
    db_dataset = configs.dataset(config, filename='database_' + str(n_labeled_samples) + '.txt', transform_mode='test')
    config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

    logging.info(f'Number of DB data: {len(db_dataset)}')
    logging.info(f'Number of Train data: {len(train_dataset)}')
    logging.info(f'Number of Test data: {len(test_dataset)}')

    train_loader = configs.dataloader(train_dataset, config['batch_size'], transform_mode='train')
    test_loader = configs.dataloader(test_dataset, config['batch_size'], workers=0, shuffle=False, drop_last=False,
                                     transform_mode='test')
    db_loader = configs.dataloader(db_dataset, config['batch_size'], shuffle=False, drop_last=False,
                                   transform_mode='test')

    return train_loader, test_loader, db_loader


def prepare_model(config, device, codebook=None):
    logging.info('Creating Model')
    model = configs.arch(config, codebook=codebook)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model


def main(config):
    device = torch.device(config.get('device'))

    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit']

    logging.info(f'Total Bit: {nbit}')
    if config['codebook_generation'] == 'N':  # normal
        codebook = torch.randn(nclass, nbit)
    elif config['codebook_generation'] == 'B':  # bernoulli
        prob = torch.ones(nclass, nbit) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
    else:  # O: optim
        codebook = get_codebook(nclass, nbit)

    # codebook = codebook.sign().to(device)
    codebook = codebook.to(device)
    io.fast_save(codebook, f'{logdir}/outputs/codebook.pth')

    train_loader, test_loader, db_loader = prepare_dataloader(config)
    model = prepare_model(config, device, codebook)

    backbone_lr_scale = 0.1
    optimizer = configs.optimizer(config, [{'params': model.parameters(),
                                            'lr': config['optim_kwargs']['lr'] * backbone_lr_scale}, ])
    scheduler = configs.scheduler(config, optimizer)

    train_history = []
    test_history = []

    loss_param = config.copy()
    loss_param.update({'device': device})

    best = 1
    best_mAP = 0
    best_recall = 0
    best_eod = 1
    best_eop = 1
    best_dp = 1
    # curr_metric = 0

    nepochs = config['epochs']
    neval = config['eval_interval']

    logging.info('Training Start')

    nbits = nbit
    pred_net_sa = AdversarialNetwork(nbits * nclass, nbits)
    optim_sa = optim.SGD(pred_net_sa.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_lambda = lambda step: 0.1 ** (step // 40)
    scheduler_sa = optim.lr_scheduler.LambdaLR(optim_sa, lr_lambda=lr_lambda, last_epoch=-1)
    # scheduler_sa = optim.lr_scheduler.StepLR(optim_sa,step_size=10,gamma = 0.1)

    for ep in range(nepochs):
        logging.info(f'Epoch [{ep + 1}/{nepochs}]')
        res = {'ep': ep + 1}

        train_meters = train_hashing(optimizer, model, optim_sa, pred_net_sa, codebook, train_loader, loss_param, config)
        scheduler.step()
        scheduler_sa.step()

        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)
        # train_outputs.append(train_out)

        eval_now = (ep + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            test_meters, test_out = test_hashing(model, codebook, test_loader, loss_param, False, True, config)
            db_meters, db_out = test_hashing(model, codebook, db_loader, loss_param, True, True, config)

            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'])

            res['recall'] = calculate_recall(db_out['codes'], db_out['labels'],
                                             test_out['codes'], test_out['labels'],
                                             loss_param['R'])

            res['odds1'], res['odds2'], res['opportunities'], res['dp'] = calculate_fairness(config, db_out['codes'],
                                                                                             db_out['labels'],
                                                                                             test_out['codes'],
                                                                                             test_out['labels'],
                                                                                             loss_param['R'], 0)

            logging.info(f'mAP: {res["mAP"]:.6f}')
            # logging.info(f'EO: {res["opportunities"]:.6f}')

            curr_metric = [res['mAP'], res['odds1'], res['opportunities'], res['dp'], res['recall']]
            if curr_metric[0] > best_mAP:
                best_mAP = curr_metric[0]
            if curr_metric[1] < best_eod:
                best_eod = curr_metric[1]
            if curr_metric[2] < best_eop:
                best_eop = curr_metric[2]
            if curr_metric[3] < best_dp:
                best_dp = curr_metric[3]
            if curr_metric[4] > best_recall:
                best_recall = curr_metric[4]
            res['best_mAP'] = best_mAP
            res['best_eod'] = best_eod
            res['best_eop'] = best_eop
            res['best_dp'] = best_dp
            res['best_recall'] = best_recall
            test_history.append(res)
            # test_outputs.append(outs)

        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            io.fast_save(db_out['codes'], f'{logdir}/outputs/db_codes.pth')
            io.fast_save(db_out['labels'], f'{logdir}/outputs/db_labels.pth')
            io.fast_save(db_out['sa_labels'], f'{logdir}/outputs/db_sa_labels.pth')

        # modelsd = model.state_dict()
        # optimsd = optimizer.state_dict()
        # io.fast_save(modelsd, f'{logdir}/models/last.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
        # save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        # if save_now:
        #     io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        # if best > curr_metric[3] and res['mAP'] > 0.55 and res['odds1'] < 0.111 and res['opportunities'] < 0.0712 and  res['dp'] < 0.066:

        # if best > curr_metric[2]: #  and res['mAP'] > 0.55
        #     best = curr_metric[2]
        #     best_mAP=curr_metric[0]
        #     best_eod=curr_metric[1]
        #     best_eop=curr_metric[2]
        #     best_dp=curr_metric[3]
        #     io.fast_save(modelsd, f'{logdir}/models/best.pth')

    # modelsd = model.state_dict()
    # io.fast_save(modelsd, f'{logdir}/models/last.pth')
    total_time = time.time() - start_time
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best_mAP * 100:.2f}')
    logging.info(f'Best eod: {best_eod * 100:.2f}')
    logging.info(f'Best eop: {best_eop * 100:.2f}')
    logging.info(f'Best dp: {best_dp * 100:.2f}')
    logging.info(f'Best recall: {best_recall * 100:.2f}')
    logging.info(f'Done: {logdir}')

    return logdir
