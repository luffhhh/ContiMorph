import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse, SwinTransformerBlock
#from blocks import PSA
from networks.dit import DiT 


class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim: int):
    super().__init__()

    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

    return embeddings

class Time_Conti_Lagrangian_flow(nn.Module):
    def __init__(self, vol_size):
        """
        Instiatiate Lagrangian_flow layer
            :param vol_size: volume size of the atlas
        """
        super(Time_Conti_Lagrangian_flow, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)
    
    
    def forward(self, inf_flow, flow2, forward_flow=True):
        shape = inf_flow.shape
        seq_len = shape[0]
        lag_flow = torch.zeros(shape, device=inf_flow.device)
        lag_flow2 = torch.zeros(shape, device=inf_flow.device)
        lag_flow[0, ::] = inf_flow[0,::]
        for k in range (1, seq_len):
            if forward_flow:
                src = lag_flow[k-1, ::].clone()
                sum_flow = inf_flow[k:k+1, ::]
                sum_flow2 = flow2[k:k+1, ::]
            else:
                src = inf_flow[k, ::]
                sum_flow = lag_flow[k-1:k, ::]

            src_x = src[0, ::]
            src_x = src_x.unsqueeze(0)
            src_x = src_x.unsqueeze(0)
            src_y = src[1, ::]
            src_y = src_y.unsqueeze(0)
            src_y = src_y.unsqueeze(0)
            
            lag_flow_x2 = self.spatial_transform(src_x, sum_flow2)
            lag_flow_y2 = self.spatial_transform(src_y, sum_flow2)
            lag_flow2[k, ::] = sum_flow2 + torch.cat((lag_flow_x2, lag_flow_y2), dim=1)

            
            lag_flow_x = self.spatial_transform(src_x, sum_flow)
            lag_flow_y = self.spatial_transform(src_y, sum_flow)
            lag_flow[k, ::] = sum_flow + torch.cat((lag_flow_x, lag_flow_y), dim=1)
        return lag_flow, lag_flow2

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        
class Block2d(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False, gse=False):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        if time_emb_dim > 0:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # up-sampling (decoder part)
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        else:
            # down-sampling (encoder part)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)    
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
    

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(F.silu(self.conv1(x)))
        time_emb = 0
        if self.time_emb_dim > 0:
            # Time embedding
            time_emb = self.time_mlp(t)
            # Extend last 2 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(F.silu(self.conv2(h)))
        # Down or Upsample
        out = F.silu(self.transform(h))

        return out


class UNet2dTimeEmb(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, in_channels=1, out_channels=1,
                 down_channels=[32, 32, 32],
                 up_channels=[32, 32, 32],
                 time_emb_dim=32) -> None:
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block2d(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block2d(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)
        ])

        self.conv1 = nn.Conv2d(up_channels[-1], up_channels[-1]//2, 3, padding=1)

    def forward(self, x, time):
        # Embed time
        t = self.time_mlp(time)

        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        x = self.conv1(x)
        return x

       

class ContiMorphNet(nn.Module):
    """
    [lagrangian_motion_estimate_net] is a class representing the architecture for Lagrangian motion estimation on a time
     sequence which is based on probabilistic diffeomoprhic VoxelMorph with a full sequence of lagrangian motion
     constraints. You may need to modify this code (e.g., number of layers) to suit your project needs.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, depth=2, time_emb_dim=64):
        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
        """
        super(ContiMorphNet, self).__init__()

        dim = len(vol_size)
        self.unet_model = UNet2dTimeEmb(in_channels=dim,
                                        out_channels=dec_nf[-1],
                                        down_channels=enc_nf,
                                        up_channels=dec_nf,
                                        time_emb_dim=time_emb_dim)
        self.dit = DiT(input_size=160,
                        patch_size=8,
                        in_channels=dec_nf[-1]//2,
                        hidden_size=dec_nf[-1],
                        depth=depth,
                        num_heads=8, learn_sigma=False)

        self.final_layer = nn.Sequential(
                    nn.Conv2d(dec_nf[-1]//2, 2, kernel_size=3, stride=1, padding=1, bias=False)
                    )
        self.spatial_transform = SpatialTransformer(vol_size)
      #  self.lag_flow = Lagrangian_flow(vol_size)
        self.lag_flow = Time_Conti_Lagrangian_flow(vol_size)
        grid = self.get_grid(vol_size)
        self.register_buffer('grid', grid)
        self.lag_regular = True

    
    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def dit_forward(self, x, t):
        x = self.dit(x, t)
        x = self.final_layer(x)
        return x
        
    def forward(self, src, tgt):
        seq_length = src.size(0)
        frame = torch.arange(seq_length).to(src)
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x, frame)
        
        t = torch.rand(seq_length).to(src)
        flow_J = t.view(seq_length, 1, 1, 1) * self.dit_forward(x, t) #frame, 2, H, W
        
        Jw = self.spatial_transform(src, flow_J)
        
        flow_I = (t - 1.).view(seq_length, 1, 1, 1) * self.dit_forward(x, t - 1.)
        Iw = self.spatial_transform(tgt, flow_I)

        image_loss = F.mse_loss(Jw, Iw)
        
        xyz = self.grid
        flow_I_J = self.compose(flow_I, flow_J, xyz)
        grid_I_J = self.make_grid(flow_I_J, xyz)
        flow_J_I = self.compose(flow_J, flow_I, xyz)
        grid_J_I = self.make_grid(flow_J_I, xyz)
        flow = (2. * t - 1.).view(seq_length, 1, 1, 1) * self.dit_forward(x, 2. * t - 1.)
        grid = self.make_grid(flow, xyz)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))
        
        t = torch.ones(seq_length).to(src)
        inf_flow = t.view(seq_length, 1, 1, 1) * self.dit_forward(x, t)
        # image warping
        y_src = self.spatial_transform(src, inf_flow)

        if self.lag_regular:
            # Lagrangian flow
            lag_flow, lag_Jw = self.lag_flow(inf_flow, flow_J)
            # Warp the reference frame by the Lagrangian flow
            src_0 = src[0, ::]
            shape = src.shape  # seq_length (batch_size), channel, height, width
            seq_length = shape[0]
            src_re = src_0.repeat(seq_length, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
            src_re = src_re.contiguous()
            lag_y_src = self.spatial_transform(src_re, lag_flow)
            lag_jw_src = self.spatial_transform(src_re, lag_Jw)
            
            lag_image_loss = F.mse_loss(Jw, lag_jw_src)
            return y_src, lag_y_src, inf_flow, lag_flow, image_loss, flow_loss, lag_image_loss
        else:
            return y_src, inf_flow, image_loss, flow_loss



    def get_grid(self, size):
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        return grid

    def make_grid(self, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow

        phi = self.grid_normalizer(phi)

        return phi

    def warp(self, image: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor, flow2: torch.Tensor, grid: torch.Tensor):
        grid = grid + flow2

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1, grid, padding_mode='reflection', align_corners=True) + flow2

        return composed_flow

    def grid_normalizer(self, grid: torch.Tensor) -> torch.Tensor:
        new_locs = grid
        shape = grid.shape[2:]
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return new_locs
