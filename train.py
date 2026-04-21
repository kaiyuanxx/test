#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import wandb
from loguru import logger
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import my_model.algorithm as algorithm
from utils.parameters import get_parameters
from utils.misc import init_wandb_workspace, setup
from dataset.owdfa_protocols import get_classes_from_protocol
from dataset.get_dataset import get_dataset
from dataset.utils.transforms import create_data_transforms


import better_exceptions
better_exceptions.hook()

args = get_parameters("/home/xky/Desktop/OW-DFA/test/config/config.yaml")
args = init_wandb_workspace(args)
if args.local_rank == 0:
    logger.add(f'{args.exam_dir}/train.log', level="INFO")
    logger.info(OmegaConf.to_yaml(args))


def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()

    # Init setup
    setup(args)

    known_classes, train_classes = get_classes_from_protocol(args.dataset.protocol)

    args.known_classes = known_classes
    args.train_classes = train_classes

    logger.info(f"Protocol {args.dataset.protocol}, with known classes {known_classes}, all classes {train_classes}")

    if args.enable_proto_pruning:
        args.model_num_classes = 10 * len(known_classes)

    # ---------------------------------------------------------
    # DATASETS & DATALOADERS
    # ---------------------------------------------------------
    # Transform
    train_transform= create_data_transforms(args.transform, 'train')
    test_transform = create_data_transforms(args.transform, 'test')

    train_dataset, test_dataset = get_dataset(args, train_transform=train_transform, test_transform=test_transform)

    # Sampler
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # Dataloader
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.train.batch_size, shuffle=False, sampler=sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=args.test.batch_size, shuffle=False, pin_memory=False)



    model = algorithm.__dict__[args.method.name](args)

    # Checkpointing: save the best model according to validation accuracy

    # Use the checkpoint directory created by `init_experiment`
    checkpoint_dir = os.path.join(args.exam_dir, 'ckpts') if args.exam_dir else None

    best_ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best_P{args.dataset.protocol}"+"_{epoch:02d}"+f"_{args.seed}",
        monitor="score", 
        mode="min",
        save_top_k=1,
        save_last=False,              
        save_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        logger = False,
        accelerator="auto",  # auto-detect GPU/CPU
        devices=1 if torch.cuda.is_available() else None,  # single-device training if CUDA available
        strategy="auto",     # no distributed strategy for single GPU
        min_epochs=1,
        max_epochs=args.train.epochs,
        default_root_dir=args.exam_dir,  # root directory for logs/checkpoints (but we set ckpt dir explicitly above)
        callbacks=[best_ckpt_cb],       # register checkpoint callback
        enable_checkpointing=True,      # must be True if you want Lightning to save checkpoints
        num_sanity_val_steps=1,         # run a few validation steps before training to catch errors
        log_every_n_steps=10,           # logging frequency
    )

    if args.eval_only:
        assert args.eval_ckpt is not None, "Please specify --eval_ckpt"

        logger.info(f"Evaluating checkpoint: {args.eval_ckpt}")

        trainer.test(
            model=model,
            dataloaders=test_dataloader,
            ckpt_path=args.eval_ckpt
        )
    else:
        trainer.fit(model, train_dataloader, test_dataloader)

    wandb.finish()



if __name__ == '__main__':
    main()