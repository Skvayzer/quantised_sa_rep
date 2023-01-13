import os
import random
import sys

import torchvision.transforms

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from argparse import ArgumentParser
import argparse
from datasets import CLEVR, CLEVRTEX, CLEVR_Mirror
from torchvision.datasets import CelebA
from models import SlotAttentionAE
import wandb
from datasets import collate_fn
from datasets import MultiDSprites

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--train_path", type=str)
program_parser.add_argument("--dataset", type=str)

# Experiment parameters
program_parser.add_argument("--device", default='gpu')
program_parser.add_argument("--batch_size", type=int, default=64)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--sa_state_dict", type=str, default='./quantised_sa_rep/clevr7_od')
program_parser.add_argument("--pretrained", type=bool, default=False)
program_parser.add_argument("--beta", type=float, default=2.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--quantization", default=True, action=argparse.BooleanOptionalAction)


# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

seed_everything(args.seed, workers=True)

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
wandb.login(key='c84312b58e94070d15277f8a5d58bb72e57be7fd')

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
dataset = args.dataset
train_dataset, val_dataset = None, None
collation = None

if dataset == 'clevr':
    train_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'train'),
                          scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_train_scenes.json'),
                          max_objs=6)

    val_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'val'),
                        scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_val_scenes.json'),
                        max_objs=6)
elif dataset == 'clevr-mirror':
    clevr_mirror = CLEVR_Mirror(images_path=os.path.join(args.train_path, 'images'),
                      scenes_path=os.path.join(args.train_path, 'scenes'),
                      max_objs=6)

    test_size = int(0.2 * len(clevr_mirror))
    train_size = len(clevr_mirror) - test_size

    train_dataset, val_dataset = torch.utils.data.random_split(clevr_mirror, [train_size, test_size])


elif dataset == 'clevr-tex':
    # TO-DO: SPECIFY THE AMOUNT OF OBJECTS LIKE DONE ABOVE
    train_dataset = CLEVRTEX(
        args.train_path, # Untar'ed
        dataset_variant='full', # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
        split='train',
        max_obj=6,
        crop=True,
        resize=(128, 128),
        return_metadata=True # Useful only for evaluation, wastes time on I/O otherwise
    )

    val_dataset = CLEVRTEX(
        args.train_path, # Untar'ed
        dataset_variant='full', # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
        split='val',
        max_obj=6,
        crop=True,
        resize=(128, 128),
        return_metadata=True # Useful only for evaluation, wastes time on I/O otherwise
    )
    collation = collate_fn
elif dataset == 'celeba':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])
    train_dataset = CelebA(root=args.train_path, split='train', target_type='attr', transform=transforms,
                           target_transform=transforms, download=False)
    val_dataset = CelebA(root=args.train_path, split='valid', target_type='attr', transform=transforms,
                         target_transform=transforms, download=False)
elif dataset == 'tetrominoes':
    train_dataset = MultiDSprites(path_to_dataset=(args.train_path + '/tetrominoes_train.npz'))
    val_dataset = MultiDSprites(path_to_dataset=(args.train_path + '/tetrominoes_val.npz'))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                          drop_last=True, collate_fn=collation)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        drop_last=True, collate_fn=collation)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("\n\nATTENTION! quantize: ", args.quantization, '\n\n', file=sys.stderr, flush=True)

# model
dict_args = vars(args)

if dataset == 'tetrominoes':
    autoencoder = SlotAttentionAE(resolution=(35, 35), hidden_size = 32, decoder_initial_size=(35, 35),
                     num_slots=4, **dict_args)
else:
    autoencoder = SlotAttentionAE(**dict_args)

project_name = 'object_detection_' + dataset + '_' + args.task

wandb_logger = WandbLogger(project=project_name, name=f'nums {args.nums!r} s {args.seed} kl {args.beta}',
                           log_model=True)
# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------


monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)
every_epoch_callback = ModelCheckpoint(every_n_epochs=10, monitor=monitor)
# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# logger_callback = SlotAttentionLogger(val_samples=next(iter(val_loader)))

callbacks = [
    checkpoint_callback,
    # logger_callback,
    every_epoch_callback,
    # swa,
    # early_stop_callback,
    lr_monitor,
]

if args.pretrained:
    state_dict = torch.load(args.sa_state_dict)
    autoencoder.load_state_dict(state_dict=state_dict, strict=False)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parameters
profiler = None  # 'simple'/'advanced'/None
accelerator = args.device
# devices = [int(args.devices)]
gpus = [0]

print(torch.cuda.device_count(), flush=True)

# trainer
trainer = pl.Trainer(accelerator=accelerator,
                     devices=[0],
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     )
#  precision=16,
# deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)
# Test
trainer.test(dataloaders=val_loader, ckpt_path=None)
wandb.finish()
