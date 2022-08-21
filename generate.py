import argparse
import os
import time
from tqdm import tqdm

import dgl
import numpy as np
import tensorboard_logger as tb_logger
import torch


from datasets import (
    NodeClassificationDataset,
)
from datasets.data_util import batcher
from models import GraphEncoder_Edge_Weighted


def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """

    model.eval()

    emb_list = []
    for idx, batch in enumerate(tqdm(train_loader)):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q.to(opt.device)
        graph_k.to(opt.device)

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def main(args_test):
    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    assert args_test.gpu is None or torch.cuda.is_available()
    print("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)


    train_dataset = NodeClassificationDataset(
        dataset=args_test.dataset,
        rw_hops=args.rw_hops,
        subgraph_size=args.subgraph_size,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
        use_pos_undirected=args.use_pos_undirected,
    )
    
    args.batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
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
        gnn_model=args.model,
        norm=args.norm,
        degree_input=args.degree_input,
        use_pos_undirected=args.use_pos_undirected
    )

    model = model.to(args.device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["pretraining_model"])

    del checkpoint

    emb = test_moco(train_loader, model, args)
    np.save(os.path.join(args.model_folder, args_test.dataset), emb.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--dataset", type=str, default="dgl",)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    main(parser.parse_args())
