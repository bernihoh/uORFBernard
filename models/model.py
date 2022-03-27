import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd


class Encoder(nn.Module):
    def __init__(self, input_nc=3, z_dim=64, bottom=False):

        super().__init__()

        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc + 4, z_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True))
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1, padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self, x):
        """
        input:
            x: input image, Bx3xHxW
        output:
            feature_map: BxCxHxW
        """
        B, C, H, W = x.shape
        X = torch.linspace(-1, 1, W)
        Y = torch.linspace(-1, 1, H)
        y1_m, x1_m = torch.meshgrid([Y, X])
        x2_m, y2_m = 2 - x1_m, 2 - y1_m  # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).type_as(x).unsqueeze(0).repeat(B, 1, 1, 1)  # Bx4xHxW
        x_ = torch.cat([x, pixel_emb], dim=1)
        
        if self.bottom:
            x_down_0 = self.enc_down_0(x_)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(x_)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map


class ShpAppEncoderV1(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.shp_out = (z_dim+(z_dim//2)) // 2
        self.app_out = (z_dim-(z_dim//2)) // 2
        self.fg_in = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                  nn.ReLU(True)) for _ in range(4)])
        self.shp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, self.shp_out),
                                                   nn.ReLU(True)) for _ in range(4)])
        self.app_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, self.app_out),
                                                   nn.ReLU(True)) for _ in range(4)])
        self.bg_in = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(True))
        self.shp_bg = nn.Sequential(nn.Linear(z_dim, self.shp_out), nn.ReLU(True))
        self.app_bg = nn.Sequential(nn.Linear(z_dim, self.app_out), nn.ReLU(True))

    def forward(self, z_slots_fg, z_slots_bg):
        z_shp_fg, z_app_fg = z_slots_fg[:, :self.shp_out].clone(), z_slots_fg[:, :self.app_out].clone()
        for i in range(len(self.fg_in)):
            z_fg_latent = self.fg_in[i](z_slots_fg[i])#.clone()
            z_shp_fg[i] = self.shp_fg[i](z_fg_latent)#.clone()
            z_app_fg[i] = self.app_fg[i](z_fg_latent)#.clone()
        z_bg_latent = self.bg_in(z_slots_bg)#.clone()
        z_shp_bg = self.shp_bg(z_bg_latent)#.clone()
        z_app_bg = self.app_bg(z_bg_latent)#.clone()
        return z_shp_fg, z_app_fg, z_shp_bg, z_app_bg


class ShpAppEncoderV2(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.shp_out = (z_dim+(z_dim//2))//2
        self.app_out = (z_dim-(z_dim//2))//2
        self.fg_in = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                  nn.ReLU(True)) for _ in range(4)])
        self.shp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, self.shp_out*2),
                                                   nn.ReLU(True)) for _ in range(4)])
        self.app_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, self.app_out*2),
                                                   nn.ReLU(True)) for _ in range(4)])
        self.max_pool = nn.MaxPool1d(2, 2)
        self.bg_in = nn.Sequential(nn.Linear(z_dim, z_dim),
                                   nn.ReLU(True))
        self.shp_bg = nn.Sequential(nn.Linear(z_dim, self.shp_out*2),
                                    nn.ReLU(True))
        self.app_bg = nn.Sequential(nn.Linear(z_dim, self.app_out*2),
                                    nn.ReLU(True))

    def forward(self, z_slots_fg, z_slots_bg):
        z_shp_fg, z_app_fg = z_slots_fg[:, :self.shp_out].clone(), z_slots_fg[:, :self.app_out].clone()
        for i in range(len(self.fg_in)):
            z_fg_latent = self.fg_in[i](z_slots_fg[i])
            z_shp_fg[i] = self.max_pool(self.shp_fg[i](z_fg_latent).unsqueeze(0))
            z_app_fg[i] = self.max_pool(self.app_fg[i](z_fg_latent).unsqueeze(0))
        z_bg_latent = self.bg_in(z_slots_bg)
        z_shp_bg = self.max_pool(self.shp_bg(z_bg_latent))
        z_app_bg = self.max_pool(self.app_bg(z_bg_latent))
        return z_shp_fg, z_app_fg, z_shp_bg, z_app_bg


class GiraffeDecoderV6(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers+1
        self.out_ch = 4
        self.z_dim = z_dim

        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_fg = nn.Sequential(nn.Linear(z_dim, z_dim),
                                            nn.ReLU(True))
        self.f_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.b_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_bg = nn.Sequential(nn.Linear(z_dim, z_dim),
                                            nn.ReLU(True))
        self.b_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        print("GiraffeDecoderV6")
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        # print("decoder")
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.z_input_shp_bg[0](z_bg_ex)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, bg_net], dim=1))
            for i in range(self.n_layers - 1):
                bg_net = bg_net + self.z_input_shp_bg[i + 1](z_bg_ex)
                bg_net = self.query_input_bg[i + 1](torch.cat([query_bg, bg_net], dim=1))
            bg_raw_shape = self.b_after_shape(bg_net).view([1, P])
            bg_net = bg_net + self.z_input_app_bg(z_bg_ex)
            bg_raw_rgb = self.b_color(torch.cat([query_bg, bg_net], dim=1)).view([1, P, 3])
            bg_raws = torch.cat([bg_raw_rgb, bg_raw_shape[..., None]], dim=-1)
            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.z_input_shp_fg[0](z_fg_ex)
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, fg_net], dim=1))
            for i in range(self.n_layers - 1):
                fg_net = fg_net + self.z_input_shp_fg[i + 1](z_fg_ex)
                fg_net = self.query_input_fg[i + 1](torch.cat([query_fg_ex, fg_net], dim=1))
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            fg_net = fg_net + self.z_input_app_fg(z_fg_ex)
            fg_raw_rgb = self.f_color(torch.cat([query_fg_ex, fg_net], dim=-1)).view(
                [K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class GiraffeDecoderV5(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers
        self.out_ch = 4
        self.z_dim = z_dim

        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_fg = nn.Sequential(nn.Linear(z_dim, z_dim),
                                            nn.ReLU(True))
        self.f_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.b_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_bg = nn.Sequential(nn.Linear(z_dim, z_dim),
                                            nn.ReLU(True))
        self.b_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.z_input_shp_bg[0](z_bg_ex)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, bg_net], dim=1))
            for i in range(self.n_layers - 1):
                bg_net = bg_net + self.z_input_shp_bg[i + 1](z_bg_ex)
                bg_net = self.query_input_bg[i + 1](torch.cat([query_bg, bg_net], dim=1))
            bg_raw_shape = self.b_after_shape(bg_net).view([1, P])
            bg_net = bg_net + self.z_input_app_bg(z_bg_ex)
            bg_raw_rgb = self.b_color(torch.cat([query_bg, bg_net], dim=1)).view([1, P, 3])
            bg_raws = torch.cat([bg_raw_rgb, bg_raw_shape[..., None]], dim=-1)
            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.z_input_shp_fg[0](z_fg_ex)
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, fg_net], dim=1))
            for i in range(self.n_layers - 1):
                fg_net = fg_net + self.z_input_shp_fg[i + 1](z_fg_ex)
                fg_net = self.query_input_fg[i + 1](torch.cat([query_fg_ex, fg_net], dim=1))
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            fg_net = fg_net + self.z_input_app_fg(z_fg_ex)
            fg_raw_rgb = self.f_color(torch.cat([query_fg_ex, fg_net], dim=-1)).view(
                [K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class GiraffeDecoderV4(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers + 1
        self.out_ch = 4
        self.input_dim = input_dim
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.out_bg = nn.Linear(z_dim, self.out_ch)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]
            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, z_bg_ex], dim=1))
            for i in range(self.n_layers - 1):
                bg_net = bg_net + self.query_input_bg[i + 1](torch.cat([query_bg, z_bg_ex], dim=1))
            bg_raws = self.out_bg(bg_net).view([1, P, self.out_ch])

            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, z_fg_ex], dim=1))
            for i in range(self.n_layers - 1):
                fg_net = fg_net + self.query_input_fg[i+1](torch.cat([query_fg_ex, z_fg_ex], dim=1))
            latent_fg = self.f_after_latent(fg_net)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class GiraffeDecoderV3(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers + 1
        self.out_ch = 4
        self.input_dim = input_dim
        query_input_dim = input_dim - z_dim
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.sharp_fg = nn.ModuleList([nn.Sequential(nn.Linear(2 * z_dim, z_dim),
                                                     nn.ReLU(True)) for _ in range(self.n_layers - 1)])
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.sharp_bg = nn.ModuleList([nn.Sequential(nn.Linear(2 * z_dim, z_dim),
                                                     nn.ReLU(True)) for _ in range(self.n_layers - 1)])
        self.out_bg = nn.Linear(z_dim, self.out_ch)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]
            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, z_bg_ex], dim=1))
            for i in range(self.n_layers - 1):
                update = self.query_input_bg[i + 1](torch.cat([query_bg, z_bg_ex], dim=1))
                bg_net = self.sharp_bg[i](torch.cat([bg_net, update], dim=1))
            bg_raws = self.out_bg(bg_net).view([1, P, self.out_ch])

            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, z_fg_ex], dim=1))
            for i in range(self.n_layers - 1):
                update = self.query_input_fg[i + 1](torch.cat([query_fg_ex, z_fg_ex], dim=1))
                fg_net = self.sharp_fg[i](torch.cat([fg_net, update], dim=1))
            latent_fg = self.f_after_latent(fg_net)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class GiraffeDecoderV2(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers
        self.out_ch = 4
        query_input_dim = input_dim - z_dim
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(query_input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_sharp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(n_layers-1)])
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(query_input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_sharp_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(n_layers-1)])
        self.out_bg = nn.Linear(z_dim, self.out_ch)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.z_input_bg[0](z_bg_ex)
            bg_net = bg_net + self.query_input_bg[0](query_bg)
            for i in range(self.n_layers - 1):
                bg_net = self.z_sharp_bg[i](bg_net)
                bg_net = bg_net + self.z_input_bg[i+1](z_bg_ex)
                bg_net = bg_net + self.query_input_bg[i+1](query_bg)

            bg_raws = self.out_bg(bg_net).view([1, P, self.out_ch])
            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.z_input_fg[0](z_fg_ex)
            fg_net = fg_net + self.query_input_fg[0](query_fg_ex)
            for i in range(self.n_layers - 1):
                fg_net = self.z_sharp_fg[i](fg_net)
                fg_net = fg_net + self.z_input_fg[i+1](z_fg_ex)
                fg_net = fg_net + self.query_input_fg[i+1](query_fg_ex)

            latent_fg = self.f_after_latent(fg_net)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4
            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class GiraffeDecoderV1(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers
        self.out_ch = 4
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])

        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])

        self.out_bg = nn.Linear(z_dim, self.out_ch)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.z_input_bg[0](z_bg_ex)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, bg_net], dim=1))
            for i in range(self.n_layers - 1):
                bg_net = bg_net + self.z_input_bg[i + 1](z_bg_ex)
                bg_net = self.query_input_bg[i+1](torch.cat([query_bg, bg_net], dim=1))

            bg_raws = self.out_bg(bg_net).view([1, P, self.out_ch])
            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.z_input_fg[0](z_fg_ex)
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, fg_net], dim=1))
            for i in range(self.n_layers - 1):
                fg_net = fg_net + self.z_input_fg[i + 1](z_fg_ex)
                fg_net = self.query_input_fg[i + 1](torch.cat([query_fg_ex, fg_net], dim=1))

            latent_fg = self.f_after_latent(fg_net)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4
            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class ShpAppDecoderV1(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers
        self.out_ch = 4
        self.z_dim = z_dim
        self.z_dim_shp = (z_dim + (z_dim // 2)) // 2
        self.z_dim_app = (z_dim - (z_dim // 2)) // 2
        query_input_dim = input_dim - z_dim
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_fg = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim_shp, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])

        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_fg = nn.Sequential(nn.Linear(self.z_dim_app, z_dim),
                                            nn.ReLU(True))
        self.f_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_shp_bg = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim_shp, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])

        # self.out_bg = nn.Linear(z_dim, self.out_ch)
        self.b_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.z_input_app_bg = nn.Sequential(nn.Linear(self.z_dim_app, z_dim),
                                            nn.ReLU(True))
        self.b_color = nn.Sequential(nn.Linear(input_dim, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, b_z_shp_fg, b_z_app_fg, b_z_shp_bg, b_z_app_bg, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        # print("decoder")
        B, K, C = b_z_shp_fg.shape[0], b_z_shp_fg.shape[1] + 1, b_z_shp_fg.shape[2] + b_z_app_fg.shape[2]
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_bg_shp = b_z_shp_bg[i]  # 1xC
            z_fg_shp = b_z_shp_fg[i]  # (K-1)xC
            z_bg_app = b_z_app_bg[i]
            z_fg_app = b_z_app_fg[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_fg_shp.shape[0] + 1, z_fg_shp.shape[1] + z_fg_app.shape[1]
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_shp_ex = z_bg_shp.expand(P, -1)
            z_bg_app_ex = z_bg_app.expand(P, -1)
            bg_net = self.z_input_shp_bg[0](z_bg_shp_ex)
            bg_net = self.query_input_bg[0](torch.cat([query_bg, bg_net], dim=1))
            for i in range(self.n_layers - 1):
                bg_net = bg_net + self.z_input_shp_bg[i + 1](z_bg_shp_ex)
                bg_net = self.query_input_bg[i + 1](torch.cat([query_bg, bg_net], dim=1))
            bg_raw_shape = self.b_after_shape(bg_net).view([1, P])
            bg_net = bg_net + self.z_input_app_bg(z_bg_app_ex)
            bg_raw_rgb = self.b_color(torch.cat([query_bg, bg_net], dim=1)).view([1, P, 3])
            bg_raws = torch.cat([bg_raw_rgb, bg_raw_shape[..., None]], dim=-1)

            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_shp_ex = z_fg_shp[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            z_fg_app_ex = z_fg_app[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)
            fg_net = self.z_input_shp_fg[0](z_fg_shp_ex)
            fg_net = self.query_input_fg[0](torch.cat([query_fg_ex, fg_net], dim=1))
            for i in range(self.n_layers - 1):
                fg_net = fg_net + self.z_input_shp_fg[i + 1](z_fg_shp_ex)
                fg_net = self.query_input_fg[i + 1](torch.cat([query_fg_ex, fg_net], dim=1))
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            fg_net = fg_net + self.z_input_app_fg(z_fg_app_ex)
            fg_raw_rgb = self.f_color(torch.cat([query_fg_ex, fg_net], dim=-1)).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class ShpAppDecoderV2(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        self.z_dim_shp = (z_dim + (z_dim // 2)) // 2
        self.z_dim_app = (z_dim - (z_dim // 2)) // 2
        input_dim_shp = input_dim - (z_dim - self.z_dim_shp)
        input_dim_app = input_dim - (z_dim - self.z_dim_app)
        before_skip = [nn.Linear(input_dim_shp, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim+input_dim_shp, z_dim), nn.ReLU(True)]
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim+input_dim_app, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim//4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim//4, 3))
        before_skip = [nn.Linear(input_dim_shp, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim_shp, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        #after_skip.append(nn.Linear(z_dim, z_dim))
        self.b_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.b_color = nn.Sequential(nn.Linear(z_dim + input_dim_app, z_dim),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, b_z_shp_fg, b_z_app_fg, b_z_shp_bg, b_z_app_bg, batched_fg_transform):
        print("ShpAppDecoderV2")
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        # print("decoder")
        B, K, C = b_z_shp_fg.shape[0], b_z_shp_fg.shape[1] + 1, b_z_shp_fg.shape[2] + b_z_app_fg.shape[2]
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_shp_bg = b_z_shp_bg[i]  # 1xC
            z_shp_fg = b_z_shp_fg[i]  # (K-1)xC
            z_app_bg = b_z_app_bg[i]
            z_app_fg = b_z_app_fg[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_shp_fg.shape[0] + 1, z_shp_fg.shape[1] + z_app_fg.shape[1]
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            # z_bg = z_slots[0:1, :]  # 1xC
            # z_fg = z_slots[1:, :]  # (K-1)xC
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            input_bg = torch.cat([query_bg, z_shp_bg.expand(P, -1)], dim=1)  # Px(60+C)
            input_bg_app = torch.cat([query_bg, z_app_bg.expand(P, -1)], dim=1)

            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_shp_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            z_fg_app_ex = z_app_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)
            input_fg_app = torch.cat([query_fg_ex, z_fg_app_ex], dim=1)

            tmp = self.b_before(input_bg)
            tmp = self.b_after(torch.cat([input_bg, tmp], dim=1))  # Px5 -> 1xPx5
            bg_raw_shape = self.b_after_shape(tmp).view([1, P])
            bg_raw_rgb = self.b_color(torch.cat([input_bg_app, tmp], dim=1)).view([1, P, self.out_ch - 1])
            bg_raws = torch.cat([bg_raw_rgb, bg_raw_shape[..., None]], dim=-1)
            tmp = self.f_before(input_fg)
            tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
            latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(torch.cat([input_fg_app, latent_fg], dim=1)).view(
                [K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
            raw_sigma = raw_masks

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class Decoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)

            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)

            tmp = self.b_before(input_bg)
            bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px5 -> 1xPx5
            tmp = self.f_before(input_fg)
            tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
            latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class ShapeAttDecoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33 + 64, z_dim=64, n_layers=3, locality=True, locality_ratio=4 / 7,
                 fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.n_layers = n_layers
        self.out_ch = 4
        query_input_dim = input_dim - z_dim
        # Foreground
        self.query_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(query_input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_sharp_fg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(n_layers-1)])
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        # Background
        self.query_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(query_input_dim, z_dim),
                                                           nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_input_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(self.n_layers)])
        self.z_sharp_bg = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, z_dim),
                                                       nn.ReLU(True)) for _ in range(n_layers-1)])
        self.out_bg = nn.Linear(z_dim, self.out_ch)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])],
                                             dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            # Background
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            z_bg_ex = z_bg.expand(P, -1)
            bg_net = self.z_input_bg[0](z_bg_ex)
            bg_net = bg_net + self.query_input_bg[0](query_bg)
            for i in range(self.n_layers - 1):
                bg_net = self.z_sharp_bg[i](bg_net)
                bg_net = bg_net + self.z_input_bg[i+1](z_bg_ex)
                bg_net = bg_net + self.query_input_bg[i+1](query_bg)

            bg_raws = self.out_bg(bg_net).view([1, P, self.out_ch])
            # Foreground
            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            fg_net = self.z_input_fg[0](z_fg_ex)
            fg_net = fg_net + self.query_input_fg[0](query_fg_ex)
            for i in range(self.n_layers - 1):
                fg_net = self.z_sharp_fg[i](fg_net)
                fg_net = fg_net + self.z_input_fg[i+1](z_fg_ex)
                fg_net = fg_net + self.query_input_fg[i+1](query_fg_ex)

            latent_fg = self.f_after_latent(fg_net)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(fg_net).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4
            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2  # <----------------------HIER ist der Colercode
            raw_sigma = raw_masks  # "<--------------------------------------- HIER ist die Dichte"

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class BadShapeAttDecoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(2 * z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(2 * z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots_shp, batched_z_slots_app, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """

        B, K, C = batched_z_slots_shp.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []
        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots_shp = batched_z_slots_shp[i]
            z_slots_app = batched_z_slots_app[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots_shp.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg_shp = z_slots_shp[0:1, :]  # 1xC
            z_fg_shp = z_slots_shp[1:, :]  # (K-1)xC
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            input_bg = torch.cat([query_bg, z_bg_shp.expand(P, -1)], dim=1)  # Px(60+C)

            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg_shp[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)

            z_bg_app = z_slots_app[0:1, :]
            z_fg_app = z_slots_app[1:, :]

            tmp = self.b_before(input_bg)

            bg_raws = self.b_after(torch.cat([input_bg, tmp, z_bg_app.expand(P, -1)], dim=1)).view(
                [1, P, self.out_ch])  # Px5 -> 1xPx5
            tmp = self.f_before(input_fg)
            tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
            z_fg_app_ex = z_fg_app[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            latent_fg = self.f_after_latent(torch.cat([tmp, z_fg_app_ex], dim=1))  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K - 1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
            fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
            raw_sigma = raw_masks

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class NeuralDecoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, channel_count=16):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.channel_count = channel_count
        self.out_ch = 4
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim+input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, int(z_dim/2)),  # z_dim//4    32  42
                                     nn.ReLU(True),
                                     nn.Linear(int(z_dim/2), self.channel_count))  # z_dim//4, 3
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.channel_count+1))  # self.out_ch instead of 33
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, batched_sampling_coor_bg, batched_sampling_coor_fg, batched_z_slots, batched_fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """

        B, K, C = batched_z_slots.shape
        _, P, _ = batched_sampling_coor_bg.shape

        raws_out = []
        masked_raws_out = []
        unmasked_raws_out = []
        masks_out = []

        # Because whole code was written for single gpu
        # This is a quick hack to take batch dim into account
        for i in range(B):
            # Get single batch
            sampling_coor_bg = batched_sampling_coor_bg[i]
            sampling_coor_fg = batched_sampling_coor_fg[i]
            z_slots = batched_z_slots[i]
            fg_transform = batched_fg_transform[i]

            K, C = z_slots.shape
            P = sampling_coor_bg.shape[0]

            if self.fixed_locality:
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
                sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
            else:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
                sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
                outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)

            sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC

            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)

            tmp = self.b_before(input_bg)
            bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.channel_count+1])  # Px33-> 1xPx33  self.out_ch instead of 33
            tmp = self.f_before(input_fg)
            tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
            latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
            fg_raw_rgb = self.f_color(latent_fg).view([K-1, P, self.channel_count])  # ((K-1)xP)x32 -> (K-1)xPx32

            fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx33

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx33
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :self.channel_count].tanh() + 1) / 2   # :, :, :3   # KxPx32
            raw_sigma = raw_masks

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx33  # [5, 1048576, 33]
            masked_raws = unmasked_raws * masks  # [5, 1048576, 33]
            raws = masked_raws.sum(dim=0)  # [1048576, 33]

            raws_out.append(raws)
            masked_raws_out.append(masked_raws)
            unmasked_raws_out.append(unmasked_raws)
            masks_out.append(masks)

        return torch.stack(raws_out), torch.stack(masked_raws_out), \
               torch.stack(unmasked_raws_out), torch.stack(masks_out)


class KmeansSlotAttention(nn.Module):  # K-means
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.num_cluster = 10  # for the k-means algorithm


        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.kmeans_bgfg = nn.Sequential(
            nn.Linear(self.num_cluster*slot_dim, (self.num_cluster + self.num_slots)//2*slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear((self.num_cluster + self.num_slots)//2 * slot_dim, self.num_slots * slot_dim),
        )

    def forward(self, feat, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        cluster_centers = self.KMeans(feat.clone().flatten(start_dim=0, end_dim=1))  # , centroids=cc_init) 10x64

        cluster_centers = cluster_centers.flatten()
        slots = self.kmeans_bgfg(cluster_centers)
        slots = slots.view((K, self.slot_dim))
        slots = slots[None, ...]
        feat_slots = self.norm_feat(torch.cat((feat, slots), dim=1))   #1x(64x64)x64  1x5x64
        feat, slots = feat_slots[:, :feat.shape[1], :], feat_slots[:, feat.shape[1]:, :]

        slot_bg, slot_fg = slots[:, 0, :][None, ...], slots[:, 1:, :]
        k = self.to_k(feat)
        v = self.to_v(feat)
        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        #print(torch.min(slots), torch.max(slots))
        return slots, attn


class SlotAttention(nn.Module):  # original
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self, feat, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters): # dot product attention
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg

            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale

            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )

            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


class YannicSlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_input = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg_input = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_input = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg_input = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_input = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg_input = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self, feat, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for i in range(self.iters+1):  # dot product attention
            if i == 0:  # input slotatt
                slot_prev_bg = slot_bg
                slot_prev_fg = slot_fg
                q_fg = self.to_q_input(slot_fg)
                q_bg = self.to_q_bg_input(slot_bg)
                dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
                dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
                dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
                attn = dots.softmax(dim=1) + self.eps  # BxKxN
                attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
                attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
                attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

                updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
                updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

                slot_bg = self.gru_bg_input(
                    updates_bg.reshape(-1, self.slot_dim),
                    slot_prev_bg.reshape(-1, self.slot_dim)
                )
                slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
                slot_bg = slot_bg + self.to_res_bg_input(slot_bg)

                slot_fg = self.gru_input(
                    updates_fg.reshape(-1, self.slot_dim),
                    slot_prev_fg.reshape(-1, self.slot_dim)
                )
                slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
                slot_fg = slot_fg + self.to_res_input(slot_fg)
            else:
                slot_prev_bg = slot_bg
                slot_prev_fg = slot_fg
                q_fg = self.to_q(slot_fg)
                q_bg = self.to_q_bg(slot_bg)

                dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
                dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
                dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
                attn = dots.softmax(dim=1) + self.eps  # BxKxN
                attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
                attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
                attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

                updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
                updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

                slot_bg = self.gru_bg(
                    updates_bg.reshape(-1, self.slot_dim),
                    slot_prev_bg.reshape(-1, self.slot_dim)
                )
                slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
                slot_bg = slot_bg + self.to_res_bg(slot_bg)

                slot_fg = self.gru(
                    updates_fg.reshape(-1, self.slot_dim),
                    slot_prev_fg.reshape(-1, self.slot_dim)
                )
                slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
                slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


class KMeansPP(nn.Module):
    def __init__(self, n_clusters, max_iter=100, tol=0.0001, return_lbl=False, device=torch.device('cuda')):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.return_lbl = return_lbl
        self.centroids = None
        self.lbl = None
        self.device = device

    def forward(self, X, centroids=None):
        self.centroids = self.centroids_init(X, centroids)
        for i in range(self.max_iter):
            centroid_added = False
            new_centroids, used_centroids = self.kmeans_step(X, self.centroids)
            centr_shift = self.calc_centr_shift(new_centroids, used_centroids)
            if new_centroids.shape[0] < self.n_clusters:
                self.centroids = self.centroids_init(X, new_centroids)
                centroid_added = True
            else:
                self.centroids = new_centroids
            if (centr_shift <= self.tol) and (not centroid_added):
                if self.return_lbl:
                    _, lbl = self.calc_dist_lbl(X, self.centroids)
                    return self.centroids, lbl
                return self.centroids
        if self.return_lbl:
            _, lbl = self.calc_dist_lbl(X, self.centroids)
            return self.centroids, lbl
        return self.centroids

    def kmeans_step(self, X, centroids):
        old_centroids = centroids
        _, lbl = self.calc_dist_lbl(X, old_centroids)
        lbl_mask, elem_per_lbl, used_lbls = self.create_lblmask_elemperlbl_usedlbl(lbl)
        x_rep = X.repeat(self.n_clusters, 1, 1)
        einsum = torch.einsum('abc,ab->abc', x_rep, lbl_mask)
        lbl_einsum_sum = torch.sum(einsum, dim=1)
        mean_sum = torch.divide(lbl_einsum_sum, elem_per_lbl)
        new_centroids = mean_sum[[~torch.any(mean_sum.isnan(), dim=1)]]
        used_centroids = old_centroids[[~torch.any(mean_sum.isnan(), dim=1)]]
        return new_centroids, used_centroids,

    def centroids_init(self, X, centroids):
        if centroids is None:
            centroids = X[torch.randint(0, X.shape[0], (1,))]
        while centroids.shape[0] < self.n_clusters:
            outlier_coor = self.calc_outlier_coor(X, centroids)
            outlier = X[outlier_coor, :][None, ...]
            centroids = torch.cat((centroids, outlier), dim=0)
        return centroids

    def calc_dist_lbl(self, X, centroids):
        sq_dist = torch.cdist(centroids, X, 2)
        min_sq_dist, lbl = torch.min(sq_dist, dim=0)
        return min_sq_dist, lbl

    def calc_outlier_coor(self, X, centroids):
        sq_dist, _ = self.calc_dist_lbl(X, centroids)
        argmax_dist = torch.argmax(sq_dist)
        return argmax_dist

    def create_lblmask_elemperlbl_usedlbl(self, lbl):
        used_lbls = torch.arange(self.n_clusters, device=self.device).view(self.n_clusters, 1)
        lbl_mask = used_lbls.repeat(1, lbl.shape[0])
        lbl_mask = torch.subtract(lbl_mask, lbl)
        lbl_mask = lbl_mask.eq(0)#.type(torch.int)
        elem_per_lbl = torch.sum(lbl_mask, dim=1).view(self.n_clusters, 1)
        return lbl_mask, elem_per_lbl, used_lbls

    def calc_centr_shift(self, centroids_1, centroids_2):
        shift = torch.subtract(centroids_1, centroids_2).abs().pow(2)
        shift = torch.sum(shift)
        return shift


def sin_emb(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=1)
    return embedded_


def raw2outputs(raw, z_vals, rays_d, render_mask=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """

    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2]).type_as(z_vals).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :3]
    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).type_as(z_vals), 1. - alpha + 1e-10], -1), -1)[:,:-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    # <----------------------- Hier den GIRAFFE Neural Renderer Einfügen
    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm


def neural_raw2outputs(raw, z_vals, rays_d, render_mask=False, channel_count = 16):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model. [16384, 64, 4]
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
        GIRAFFE: Uses the neural renderer network instead this method
    """
    #print("neural raw two outputs")

    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)  # [16384, 64]
    rgb = raw[..., :channel_count]  # [16384, 64, 32]
    alpha = raw2alpha(raw[..., channel_count], dists)  # [N_rays, N_samples] [16384, 64]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:-1] # [16384, 64]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 32] [16384, 32]
    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)  # [16384, 64]
    depth_map = torch.sum(weights_norm * z_vals, -1)  # [16384]

    if render_mask:
        density = raw[..., 16]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm


class RecursiveNeuralRenderer(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        assert channel_count == 16
        self.channel_count = channel_count
        self.input_conv = nn.Conv3d(self.channel_count, 3, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.rgb_conv = nn.ModuleList([nn.Conv3d(int(self.channel_count/2**i), 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)) for i in range(3)])
        self.rgb_conv.append(nn.Conv3d(3, 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)))

        self.conv = nn.ModuleList([nn.Sequential(nn.Conv3d(int(self.channel_count/2**i), int(self.channel_count/2**(i+1)), (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                                 nn.ReLU(inplace=True)
                                                 ) for i in range(2)])
        self.conv.append(nn.Sequential(nn.Conv3d(int(self.channel_count/2**2), 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                       nn.ReLU(inplace=True)))

    def forward(self, x):
        assert x.shape[1] == self.channel_count

        rgb_tensor = self.input_conv(x)
        for i in range(3):
            rgb_tensor = rgb_tensor + self.rgb_conv[i](x)
            x = self.conv[i](x)
        rgb_tensor = rgb_tensor + self.rgb_conv[-1](x)

        # final activation
        return torch.tanh(rgb_tensor)


class UnetNeuralRenderer(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        assert channel_count == 16
        self.cc = channel_count
        self.maxPool3D = nn.ModuleList([nn.MaxPool3d((1, 2, 2), (1, 2, 2)) for _ in range(3)])
        self.upSample3D = nn.ModuleList([nn.Upsample(scale_factor=(1, 2, 2)) for _ in range(3)])
        self.convDown = nn.ModuleList([nn.Sequential(nn.Conv3d(self.cc*2**i, self.cc*2**(i+1), (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                                     nn.ReLU(inplace=True)) for i in range(2)])
        self.convDown.append(nn.Sequential(nn.Conv3d(self.cc*2**2, self.cc*2**2, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                           nn.ReLU(inplace=True)))
        self.convUp = nn.ModuleList([nn.Sequential(nn.Conv3d(2*self.cc*2**(2-i), self.cc*2**(2-(i+1)), (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                                   nn.ReLU(inplace=True)) for i in range(2)])
        self.convUp.append(nn.Sequential(nn.Conv3d(self.cc*2, 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                                         nn.ReLU(inplace=True)))

    def forward(self, x):
        assert x.shape[1] == self.cc

        convDownResults = []
        for i in range(3):
            convDownResults.append(x)
            x = self.maxPool3D[i](x)
            x = self.convDown[i](x)
        for i in range(3):
            x = self.upSample3D[i](x)
            x = torch.concat([convDownResults[2-i], x], dim=1)
            x = self.convUp[i](x)
        return torch.tanh(x)


def get_perceptual_net(layer=4):
    assert layer > 0
    idx_set = [None, 4, 9, 16, 23, 30]
    idx = idx_set[layer]
    vgg = vgg16(pretrained=True)
    loss_network = nn.Sequential(*list(vgg.features)[:idx]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    return loss_network


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        stride=1,
        padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, ndf, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf*2,
            8: ndf*2,
            16: ndf,
            32: ndf,
            64: ndf//2,
            128: ndf//2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

