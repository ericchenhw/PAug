import argparse
import copy
from email.mime import base
from importlib.resources import path
import os
import time
import warnings


import dgl
import numpy as np
import torch.nn.functional as F 
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from utils.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from datasets import (
    LoadBalanceGraphDataset,
    LinkPredictionDatasetLabeled, 
    worker_init_fn,
)
from datasets.data_util import batcher, labeled_batcher, link_prediction_batcher
from datasets.graph_dataset import MultipleLoadBalanceGraphDataset
from models import GraphEncoder_Edge_Weighted
from utils.misc import AverageMeter, adjust_learning_rate, warmup_linear
from models import Graph_Augmentation_Model
from utils.reg_regularizer import cal_reg


from utils.early_stopping import EarlyStopping

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=10, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")

    # optimization
    # parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    parser.add_argument("--perturbation_strength", type=float, required=True, help='the coefficient of the regularization on the perturbation bound, beta') 
    parser.add_argument("--aug_temperature", type=float, default=10.0)
    parser.add_argument("--adversarial_strength", type=float, default=1.0) 

    parser.add_argument("--perturbation_threshold", type=float, required=True, help='the margin (m) in the perturbation bound loss')
    parser.add_argument("--pretraining_dataset", type=str, default='h-index_mst_twin_domain1')
    parser.add_argument("--adversarial_dataset", type=str, default='h-index_mst_twin_domain1')
    parser.add_argument("--use_pos_undirected", action="store_true", default=False)
    parser.add_argument("--degree_input", action="store_true", default=False)
    parser.add_argument("--learning_rate_aug", type=float, default=5e-3)
    parser.add_argument("--semantic_strength", type=float, default=1.0, help='the coefficient of the regularization on the semantic bound, gamma') 
    parser.add_argument("--iterative_contrastive_interval", type=int, default=10)
    parser.add_argument("--max_patience_discriminator", type=int, default=10)
    parser.add_argument("--max_patience_aug", type=int, default=15)
    parser.add_argument("--max_patience_contrastive", type=int, default=10)
    parser.add_argument("--adversarial_repeat_times", type=int, default=10)
    parser.add_argument("--idle_interval", type=int, default=20, help='the fist idle_interval epochs used to train the encoder')

    # # dataset definition
    # parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin_edge_weighted", choices=["gin_edge_weighted"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # # finetune setting
    # parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on


    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_data_{}_adv_data_{}_{}_perturb_stre_{}_perturb_thre_{}_semantic_str_{}_adver_stre_{}_idle_int{}_pat_dis_{}_pat_aug_{}_pat_con_{}".format(
        opt.exp,
        opt.pretraining_dataset,
        opt.adversarial_dataset,
        opt.model,
        opt.perturbation_strength,
        opt.perturbation_threshold,
        opt.semantic_strength,
        opt.adversarial_strength,
        opt.idle_interval,
        opt.max_patience_discriminator,
        opt.max_patience_aug,
        opt.max_patience_contrastive,      
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt




def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def train_contrastive_loss(
    epoch, train_loader, model, model_aug, criterion, optimizer_model, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    
    early_stopping = EarlyStopping(None, patience=50)
    
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch

        model.train()
        model_aug.eval()

        graph_q.to(torch.device(opt.gpu))
        graph_k.to(torch.device(opt.gpu))
        edge_weight_aug_q = model_aug(graph_q, device=opt.gpu).detach()
        edge_weight_aug_k = model_aug(graph_k, device=opt.gpu).detach()

        bsz = graph_q.batch_size

        feat_q = model(graph_q, edge_weight=edge_weight_aug_q)
        feat_k = model(graph_k, edge_weight=edge_weight_aug_k)

        out = torch.matmul(feat_q, feat_q.t()) / opt.nce_t
        out_new = torch.matmul(feat_k, feat_q.t()) / opt.nce_t
        probs = out_new[range(graph_q.batch_size), range(graph_q.batch_size)]
        out[range(graph_q.batch_size), range(graph_q.batch_size)] = probs
        prob = probs.mean()
        # prob = out_new[range(graph_q.batch_size), range(graph_q.batch_size)].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer_model.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        # global_step = epoch * n_batch + idx
        # lr_this_step = opt.learning_rate * warmup_linear(
        #     global_step / (opt.epochs * n_batch), 0.1
        # )
        # for param_group in optimizer_model.param_groups:
        #     param_group["lr"] = lr_this_step
        optimizer_model.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            graph_q.number_of_nodes() / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train_Encoder: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global tb_step
            tb_step += 1
            # global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, tb_step)
            sw.add_scalar("moco_prob", prob_meter.avg, tb_step)
            sw.add_scalar("graph_size", graph_size.avg, tb_step)
            sw.add_scalar("graph_size/max", max_num_nodes, tb_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, tb_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, tb_step)
            sw.add_scalar("learning_rate", optimizer_model.param_groups[0]["lr"], tb_step)

            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()


            max_num_nodes, max_num_edges = 0, 0 

        early_stopping(loss.item(), None)
        if early_stopping.early_stop:
            print("early stopped")
            return False
    return True
                    
    

def train_link_predictor(
    epoch, adversarial_loader, model, model_aug, link_predictor, optimizer_link_predictor, sw, opt  
):
    n_batch = adversarial_loader.dataset.total // opt.batch_size
    
    link_predictor_performance_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    gnorm_meter = AverageMeter()

    early_stopping = EarlyStopping(None, patience=100)    
    
    for idx, batch in enumerate(adversarial_loader):
        g1, g2, y = batch

        g1.to(torch.device(opt.gpu))
        g2.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))
        bsz = g1.batch_size

        model.eval()
        model_aug.eval()
        # if model_aug is not None:
        #     model_aug.eval()
        link_predictor.train()
        

        edge_weight_aug1 = model_aug(g1, device=opt.gpu).detach() # if model_aug is not None else None
        feat1 = model(g1, edge_weight=edge_weight_aug1)
        edge_weight_aug2 = model_aug(g2, device=opt.gpu).detach() # if model_aug is not None else None
        feat2 = model(g2, edge_weight=edge_weight_aug2)
        
        x = torch.cat([feat1, feat2], dim=-1)
        out = link_predictor(x).squeeze()
        
        adv_loss = -1 * (y * torch.log(out + 1e-10) + (1-y) * torch.log(1-out + 1e-10)).mean()
        loss = opt.adversarial_strength * adv_loss
        
        optimizer_link_predictor.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(link_predictor.parameters(), opt.clip_norm)
        
        # global_step = epoch * n_batch + idx
        # lr_this_step = opt.learning_rate_aug * warmup_linear(
        #     global_step / (opt.epochs * n_batch), 0.1
        # )
        # for param_group in optimizer_link_predictor.param_groups:
        #     param_group["lr"] = lr_this_step
        
        optimizer_link_predictor.step()

        out = out.detach().cpu()
        auc = roc_auc_score(y.cpu(), out)
        link_predictor_performance_meter.update(auc, bsz)
        adv_loss_meter.update(adv_loss.item(), bsz)
        gnorm_meter.update(grad_norm.item(), 1)

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global tb_step
            tb_step += 1
            sw.add_scalar("adv_loss_when_training_link_predictor", adv_loss_meter.avg, tb_step)
            sw.add_scalar("link_predictor_performance(AUC)", link_predictor_performance_meter.avg, tb_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, tb_step)

            gnorm_meter.reset()
            adv_loss_meter.reset()
            link_predictor_performance_meter.reset()

        
        early_stopping(adv_loss.item(), None)
        if early_stopping.early_stop:
            return True
    return False
        

def train_aug_loss(
    epoch, adversarial_loader, model, model_aug, link_predictor, aux_loader, optimizer_aug, optimizer_link_predictor, sw, opt
):
    """
    one epoch training for each loss seperately
    """
    n_batch = adversarial_loader.dataset.total // opt.batch_size

    batch_time = AverageMeter()
    data_time = AverageMeter()

    reg_loss_meter = AverageMeter()
    edge_drop_percentage_meter = AverageMeter()
    adv_loss_when_training_aug_meter = AverageMeter()


    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0
    
    aux_iter = iter(aux_loader)
      
    early_stopping = EarlyStopping(None, patience=100)
    
    for idx, batch in enumerate(adversarial_loader):
        ## ================== train adversarially and reg_loss===================
        ## if perturbation_strength <= 0, then skip reg loss
        g1, g2, y = batch

        g1.to(torch.device(opt.gpu))
        g2.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))
        bsz = g1.batch_size
        
        model.eval() 
        model_aug.train()
        link_predictor.eval()

        edge_weight_aug1 = model_aug(g1, device=opt.gpu)
        feat1 = model(g1, edge_weight=edge_weight_aug1)
        edge_weight_aug2 = model_aug(g2, device=opt.gpu)
        feat2 = model(g2, edge_weight=edge_weight_aug2)
        
        x = torch.cat([feat1, feat2], dim=-1)
        out = link_predictor(x).squeeze()
        
        adv_loss = -1 * (y * torch.log(out + 1e-10) + (1-y) * torch.log(1-out + 1e-10)).mean()
        if opt.perturbation_strength > 0:
            edge_weight_soft = model_aug(g1, device=opt.gpu, return_soft=True)
            reg_loss = cal_reg(g1, edge_weight_soft, threshold=opt.perturbation_threshold, device=opt.gpu, squared=True)
        else:
            reg_loss = torch.tensor([0.0], dtype=torch.float32).to(opt.gpu)
        edge_drop_percentage = cal_reg(g1, edge_weight_aug1, threshold=0, device=opt.gpu)
        
        loss = -1 * opt.adversarial_strength * adv_loss + opt.perturbation_strength * reg_loss
        
        optimizer_aug.zero_grad()
        loss.backward()
        # grad_norm = clip_grad_norm(model_aug.parameters(), opt.clip_norm)
        # global_step = epoch * n_batch + idx
        # lr_this_step = opt.learning_rate_aug * warmup_linear(
        #     global_step / (opt.epochs * n_batch), 0.1
        # )
        # for param_group in optimizer_aug.param_groups:
        #     param_group["lr"] = lr_this_step
            
        optimizer_aug.step()
            
        reg_loss_meter.update(reg_loss.item(), bsz)
        adv_loss_when_training_aug_meter.update(adv_loss.item(), bsz)
        edge_drop_percentage_meter.update(edge_drop_percentage.item(), bsz)
        
        # ======================= train aux ===========================================
        if args.semantic_strength > 0:
            try:
                batch = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader)
                batch = next(aux_iter)
            graphs = batch
            for g in graphs:
                g.to(torch.device(opt.gpu))
            
            model.eval() 
            model_aug.train()

            ori_graphs = graphs
            aug_graphs = graphs
            
            feats_ori = [model(g) for g in ori_graphs]
            feats_aug = [model(g, edge_weight=model_aug(g, device=opt.gpu)) for g in aug_graphs]
            
            feats_ori = torch.stack(feats_ori, dim=-1).mean(dim=-1, keepdim=False)
            feats_aug = torch.stack(feats_aug, dim=-1).mean(dim=-1, keepdim=False)
            
            dist = feats_ori - feats_aug
            dist = (dist * dist).sum(-1)
            dist = torch.exp(dist).mean()
            
            loss = opt.semantic_strength * dist
            
            optimizer_aug.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model_aug.parameters(), opt.clip_norm)
            global_step = epoch * n_batch + idx
            # lr_this_step = opt.learning_rate_aug * warmup_linear(
            #     global_step / (opt.epochs * n_batch), 0.1
            # )
            # for param_group in optimizer_aug.param_groups:
            #     param_group["lr"] = lr_this_step
            
            aux_loss_meter.update(dist.item(), bsz)
            gnorm_meter.update(grad_norm.item(), 1)
            
        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print("Train_Aug: [{0}][{1}/{2}]\t"
                  "reg_loss {reg_loss.val:.3f}({reg_loss.avg:.3f})\t"
                  "adv_loss {adv_loss.val:.3f}({adv_loss.avg:.3f})\t"
                  "edge_drop_percentage {edge_drop_percentage.val:.3f} ({edge_drop_percentage.avg:.3f})"
                  "aux_loss {aux_loss.val:.3f} ({aux_loss.avg:.3f})\t".format(
                      epoch,
                      idx+1,
                      n_batch,
                      reg_loss = reg_loss_meter,
                      adv_loss = adv_loss_when_training_aug_meter,
                      edge_drop_percentage = edge_drop_percentage_meter,
                      aux_loss = aux_loss_meter,
                  )
                  )
            
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global tb_step
            tb_step += 1

            sw.add_scalar("graph_size", graph_size.avg, tb_step)
            sw.add_scalar("graph_size/max", max_num_nodes, tb_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, tb_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, tb_step)
            # sw.add_scalar("learning_rate_aug", optimizer_aug.param_groups[0]["lr"], tb_step)
            
            sw.add_scalar("adv_loss_when_training_aug", adv_loss_when_training_aug_meter.avg, tb_step)
            sw.add_scalar("reg_loss", reg_loss_meter.avg, tb_step)
            sw.add_scalar("edge_drop_percentage", edge_drop_percentage_meter.avg, tb_step)
            sw.add_scalar("aux_loss", aux_loss_meter.avg, tb_step)

            graph_size.reset()
            gnorm_meter.reset()

            adv_loss_when_training_aug_meter.reset()
            reg_loss_meter.reset()
            edge_drop_percentage_meter.reset()
            aux_loss_meter.reset()
            max_num_nodes, max_num_edges = 0, 0 

        early_stopping(loss.item(), None)
        if early_stopping.early_stop:
            print("early stopped")
            return True
        
    return False
            
        
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    
    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)

    train_dataset = LoadBalanceGraphDataset(
        rw_hops=args.rw_hops,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        dgl_graphs_file='data/' + args.pretraining_dataset + '.bin',
        num_copies=args.num_copies,
        use_pos_undirected=args.use_pos_undirected
    )
    
    train_loader = torch.utils.data.DataLoader( 
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    adversarial_dataset = LinkPredictionDatasetLabeled(
        dataset=args.adversarial_dataset,
        rw_hops=args.rw_hops,
        subgraph_size=args.subgraph_size,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
        use_pos_undirected=args.use_pos_undirected
    )

    adversarial_loader = torch.utils.data.DataLoader(
        dataset=adversarial_dataset,
        batch_size=args.batch_size,
        collate_fn=link_prediction_batcher(),
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    aux_dataset = MultipleLoadBalanceGraphDataset(
        rw_hops=args.rw_hops,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        dgl_graphs_file='data/' + args.pretraining_dataset + '.bin',
        num_copies=args.num_copies,
        use_pos_undirected=args.use_pos_undirected,
        rwr_num_per_node=2,
    )
    
    aux_loader = torch.utils.data.DataLoader(
                dataset=aux_dataset,
                batch_size=args.batch_size,
                collate_fn=batcher(),
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=worker_init_fn,
            )

    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)
    
    criterion = NCESoftmaxLossNS()
    criterion = criterion.cuda(args.gpu)

    model = GraphEncoder_Edge_Weighted(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=args.degree_input,
            use_pos_undirected=args.use_pos_undirected
        )


    model = model.cuda(args.gpu)

    link_predictor = torch.nn.Sequential(nn.Linear(2*args.hidden_size, args.hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(args.hidden_size, 1),
                                        nn.Sigmoid(),).cuda(args.gpu)


    optimizer_model = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    optimizer_link_predictor = torch.optim.Adam(
        link_predictor.parameters(),
        lr=args.learning_rate_aug,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    model_aug = Graph_Augmentation_Model(        
        GraphEncoder_Edge_Weighted(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=args.degree_input,
            use_pos_undirected=args.use_pos_undirected
        ).cuda(args.gpu),
        args.hidden_size * (args.num_layer - 1),
        64,
        args.aug_temperature
    ).cuda(args.gpu)

    optimizer_aug = torch.optim.Adam(
        model_aug.parameters(),
        lr=args.learning_rate_aug,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        model.load_state_dict(checkpoint["model"])

        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        del checkpoint
        torch.cuda.empty_cache()

    sw = SummaryWriter(args.tb_folder)

    global tb_step, link_pred_step
    tb_step, link_pred_step = 0, 0
    train_discriminator, train_aug, train_contrastive = True, False, False
    iter_discriminator, iter_aug, iter_contrastive = 0, 0, 0
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        
        print("==> training...")

        time1 = time.time()

        if epoch <= args.idle_interval:
            loss = train_contrastive_loss(
                epoch,
                train_loader,
                model,
                model_aug,
                criterion,
                optimizer_model,
                sw,
                args,
            )
        else:
            for r in range(args.adversarial_repeat_times):
                while train_discriminator:
                    iter_discriminator += 1
                    early_stopped = train_link_predictor(epoch,
                                                         adversarial_loader,
                                                         model,
                                                         model_aug,
                                                         link_predictor,
                                                         optimizer_link_predictor,
                                                         sw,
                                                         args)
                    if early_stopped or iter_discriminator > args.max_patience_discriminator:
                        iter_discriminator = 0
                        train_aug = True
                        train_discriminator = False
                
                while train_aug:
                    iter_aug += 1
                    early_stopped = train_aug_loss(
                        epoch,
                        adversarial_loader,
                        model,
                        model_aug, 
                        link_predictor, 
                        aux_loader,
                        optimizer_aug, 
                        optimizer_link_predictor,
                        sw,
                        args
                    )
                    if early_stopped or iter_aug > args.max_patience_aug:
                        iter_aug = 0
                        train_aug = False
                        train_discriminator = True

            train_contrastive = True
            while train_contrastive:
                iter_contrastive += 1
                early_stopped = train_contrastive_loss(
                                                    epoch,
                                                    train_loader,
                                                    model,
                                                    model_aug,
                                                    criterion,
                                                    optimizer_model,
                                                    sw,
                                                    args,
                                                )
                if iter_contrastive > args.max_patience_contrastive:
                    iter_contrastive=0
                    train_contrastive=False
                    
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "optimizer_model": optimizer_model.state_dict(),
                "epoch": epoch,
            }
            state['model_aug'] = model_aug.state_dict()
            state['optimizer_aug'] = optimizer_aug.state_dict()
            state['link_predictor'] = link_predictor.state_dict()
            
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state
        
        
        # saving the model
        print("==> Saving...")
        state = {
            "opt": args,
            "model": model.state_dict(),
            "optimizer_model": optimizer_model.state_dict(),
            "epoch": epoch,
        }

        state['model_aug'] = model_aug.state_dict()
        state['optimizer_aug'] = optimizer_aug.state_dict()
        state['link_predictor'] = link_predictor.state_dict()

        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":

    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    args.gpu = args.gpu[0]
    main(args)