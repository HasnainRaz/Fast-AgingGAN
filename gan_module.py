import itertools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dataset import ImagetoImageDataset
from models import FastGenerator, Discriminator, RhoClipper


class AgingGAN(pl.LightningModule):

    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.hparams = hparams
        self.genA2B = FastGenerator(hparams['ngf'], n_blocks=hparams['n_blocks'])
        self.genB2A = FastGenerator(hparams['ngf'], n_blocks=hparams['n_blocks'])
        self.disGA = Discriminator(hparams['ndf'], hparams['n_layers'])
        self.disGB = Discriminator(hparams['ndf'], hparams['n_layers'])
        self.disLA = Discriminator(hparams['ndf'], hparams['n_layers'] - 2)
        self.disLB = Discriminator(hparams['ndf'], hparams['n_layers'] - 2)

        self.Rho_clipper = RhoClipper(0, 1)

        # cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        return self.genA2B(x)

    def training_step(self, batch, batch_idxs, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = F.mse_loss(real_GA_logit, torch.ones_like(real_GA_logit)) + \
                           F.mse_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
            D_ad_cam_loss_GA = F.mse_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)) + \
                               F.mse_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = F.mse_loss(real_LA_logit, torch.ones_like(real_LA_logit)) + \
                           F.mse_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
            D_ad_cam_loss_LA = F.mse_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)) + \
                               F.mse_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = F.mse_loss(real_GB_logit, torch.ones_like(real_GB_logit)) + \
                           F.mse_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
            D_ad_cam_loss_GB = F.mse_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)) + \
                               F.mse_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = F.mse_loss(real_LB_logit, torch.ones_like(real_LB_logit)) + \
                           F.mse_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))
            D_ad_cam_loss_LB = F.mse_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)) + \
                               F.mse_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))

            D_loss_A = self.hparams['adv_weight'] * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.hparams['adv_weight'] * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            d_loss = D_loss_A + D_loss_B
            output = {
                'loss': d_loss,
                'log': {'Loss/Discriminator': d_loss}
            }
            return output

        if optimizer_idx == 1:
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = F.mse_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = F.mse_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = F.mse_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = F.mse_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = F.mse_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = F.mse_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = F.mse_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = F.mse_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit))

            G_recon_loss_A = F.l1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = F.l1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = F.l1_loss(fake_A2A, real_A)
            G_identity_loss_B = F.l1_loss(fake_B2B, real_B)

            G_cam_loss_A = F.binary_cross_entropy_with_logits(fake_B2A_cam_logit, torch.ones_like(
                fake_B2A_cam_logit)) + F.binary_cross_entropy_with_logits(fake_A2A_cam_logit,
                                                                          torch.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = F.binary_cross_entropy_with_logits(fake_A2B_cam_logit, torch.ones_like(
                fake_A2B_cam_logit)) + F.binary_cross_entropy_with_logits(fake_B2B_cam_logit,
                                                                          torch.zeros_like(fake_B2B_cam_logit))

            G_loss_A = self.hparams['adv_weight'] * (
                        G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                       self.hparams['cycle_weight'] * G_recon_loss_A + self.hparams[
                           'identity_weight'] * G_identity_loss_A + \
                       self.hparams['cam_weight'] * G_cam_loss_A
            G_loss_B = self.hparams['adv_weight'] * (
                        G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                       self.hparams['cycle_weight'] * G_recon_loss_B + self.hparams[
                           'identity_weight'] * G_identity_loss_B + \
                       self.hparams['cam_weight'] * G_cam_loss_B

            g_loss = G_loss_A + G_loss_B
            output = {
                'loss': g_loss,
                'log': {'Loss/Generator': g_loss}
            }
            self.generated_B = fake_A2B
            self.generated_A = fake_B2A

            self.real_B = real_B
            self.real_A = real_A
            return output

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters(),
                                                   self.disLA.parameters(),
                                                   self.disLB.parameters()),
                                   lr=self.hparams['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [d_optim, g_optim], []

    def on_epoch_end(self):
        self.logger.experiment.add_image('Real/A', make_grid(self.real_A, normalize=True, scale_each=True),
                                         self.current_epoch)
        self.logger.experiment.add_image('Real/B', make_grid(self.real_B, normalize=True, scale_each=True),
                                         self.current_epoch)
        self.logger.experiment.add_image('Generated/A', make_grid(self.generated_A, normalize=True, scale_each=True),
                                         self.current_epoch)
        self.logger.experiment.add_image('Generated/B', make_grid(self.generated_B, normalize=True, scale_each=True),
                                         self.current_epoch)

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 30, self.hparams['img_size'] + 30)),
            transforms.RandomCrop(self.hparams['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)
