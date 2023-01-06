import os
import sys

import pytorch_lightning as pl
import torch
import wandb

from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, CoordQuantizer
from modules.slot_attention import SlotAttentionBase
from utils import spatial_broadcast, spatial_flatten, adjusted_rand_index

class SlotAttentionAE(pl.LightningModule):
    """
    Slot attention based autoencoder for object discovery dataset
    """

    def __init__(self,
                 resolution=(128, 128),
                 num_slots=7,
                 num_iters=3,
                 in_channels=3,
                 slot_size=64,
                 hidden_size=64,
                 beta=2,
                 lr=4e-4,
                 dataset='',
                 nums=[8, 8, 8, 8],
                 decoder_initial_size = (8, 8),
                 num_steps=int(3e5), **kwargs
                 ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.dataset = dataset

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(3)]
        )
        self.decoder_initial_size = decoder_initial_size

        # Decoder
        self.decoder = Decoder()

        self.enc_emb = PosEmbeds(hidden_size, self.resolution)
        self.dec_emb = PosEmbeds(hidden_size, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )
        self.slots_lin = nn.Linear(hidden_size * 2, hidden_size)

        self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                hidden_dim=slot_size * 2)
        self.coord_quantizer = CoordQuantizer(nums)
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta
        self.save_hyperparameters()

    def forward(self, inputs):


        x = self.encoder(inputs)
        x = self.enc_emb(x)

        x = spatial_flatten(x[0])
        x = self.layer_norm(x)
        x = self.mlp(x)

        slots = self.slot_attention(x)

        sys.stderr.write("\nslot shape:\n " + str(slots.shape))


        props, coords, kl_loss = self.coord_quantizer(slots)
        slots = torch.cat([props, coords], dim=-1)
        sys.stderr.write("\nslot shape:\n " + str(slots.shape))
        sys.stderr.write("\nprops:\n " + str(props.shape))
        sys.stderr.write("\ncoords:\n " + str(coords.shape))


        slots = self.slots_lin(slots)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        x = self.dec_emb(x)
        x = self.decoder(x[0])

        x = x.reshape(inputs.shape[0], self.num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons, kl_loss, masks

    def step(self, batch):
        # a = 'A'*10
        # sys.stderr.write(a + "BATCH LEN" + str(len(batch)))
        # print("aaa", file=sys.stderr, flush=True)

        imgs = batch['image']
        sys.stderr.write("\nimg shape:\n " + str(imgs.shape))
        result, _, kl_loss, _ = self(imgs)
        loss = F.mse_loss(result, imgs)
        return loss, kl_loss



    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, kl_loss = self.step(batch)

        self.log('Training MSE', loss)
        self.log('Training KL', kl_loss)
        # print("TRAINING STEP: ", batch_idx, file=sys.stderr, flush=True)
        if batch_idx == 0:
            imgs = batch['image'][:8]
            result, recons, _, pred_masks = self(imgs)
            if self.dataset == 'clevr-mirror':
                self.trainer.logger.experiment.log({
                    'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
                    'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)],
                    # 'true_masks': [wandb.Image(x) for x in torch.unsqueeze(true_masks, dim=-1)],
                    # 'pred_masks': [wandb.Image(x) for x in torch.unsqueeze(pred_masks, dim=-1)]
                })
                self.trainer.logger.experiment.log({
                    f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]
                    for i in range(self.num_slots)
                })
        loss = loss + kl_loss * self.beta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log('lr', sch.get_last_lr()[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl_loss = self.step(batch)
        self.log('Validation MSE', loss)
        self.log('Validation KL', kl_loss)

        if batch_idx == 0:
            imgs = batch['image'][:8]
            # print("img: ", imgs.shape, file=sys.stderr, flush=True)
            if self.dataset == 'clevr-tex':
                true_masks = batch['mask'][:8]
            result, recons, _, pred_masks = self(imgs)
            pred_masks = torch.squeeze(pred_masks)

            # print("ATTENTION! MASKS (true/pred): ", true_masks.shape, pred_masks.shape, file=sys.stderr, flush=True)
            # print("TRUE: ", true_masks[true_masks > 0], file=sys.stderr, flush=True)
            # print("PRED: ", pred_masks, file=sys.stderr, flush=True)

            self.trainer.logger.experiment.log({
                'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
                'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)],
                # 'true_masks': [wandb.Image(x) for x in torch.unsqueeze(true_masks, dim=-1)],
                # 'pred_masks': [wandb.Image(x) for x in torch.unsqueeze(pred_masks, dim=-1)]
            })
            self.trainer.logger.experiment.log({
                f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]
                for i in range(self.num_slots)
            })

            if self.dataset == 'clevr-tex':
                pred_masks = pred_masks.view(*pred_masks.shape[:2], -1)
                true_masks = true_masks.view(*true_masks.shape[:2], -1)
                # print("ATTENTION! MASKS (true/pred): ", true_masks.shape, pred_masks.shape, file=sys.stderr, flush=True)
                self.log('ARI', adjusted_rand_index(true_masks.float().cpu(), pred_masks.float().cpu()).mean())
        return loss

    def validation_epoch_end(self, outputdata):
        if self.current_epoch % 10 == 0:
            save_path = "./sa_autoencoder_end_to_end"
            self.trainer.save_checkpoint(os.path.join(save_path, f"{self.current_epoch}_{self.beta}_{self.dataset}_sa_od_pretrained.ckpt"))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.num_steps, pct_start=0.05)
        return [optimizer], [scheduler]
