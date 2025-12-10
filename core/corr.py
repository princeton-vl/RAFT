import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())

class AltCudaCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, r):
        ctx.save_for_backward(fmap1, fmap2_i, coords)
        ctx.r = r
        corr, = alt_cuda_corr.forward(fmap1, fmap2_i, coords, r)
        return corr,
        # this should be different from return alt_cuda_corr.forward(...
    
    @staticmethod
    def backward(ctx, corr_grad):
        fmap1, fmap2_i, coords = ctx.saved_tensors
        corr_grad = corr_grad.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = alt_cuda_corr.backward(fmap1, fmap2_i, coords, corr_grad, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None
    


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = AltCudaCorr.apply(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

# --- NEW CPU-SAFE CORRELATION METHOD ---

def localized_corr(fmap1, fmap2, r):
    """
    Non-warping localized correlation implementation using standard PyTorch functions.
    This is fast and CPU-safe, suitable for inference.
    """
    B, C, H, W = fmap1.shape
    corr_channels = (2*r+1)*(2*r+1)
    corr = torch.zeros(B, corr_channels, H, W, device=fmap1.device, dtype=fmap1.dtype)

    idx = 0
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            pad_l = max(dx, 0)
            pad_r = max(-dx, 0)
            pad_t = max(dy, 0)
            pad_b = max(-dy, 0)

            shifted = F.pad(fmap2, (pad_l, pad_r, pad_t, pad_b))
            shifted = shifted[:, :, pad_t:H + pad_t, pad_l:W + pad_l] 
            
            corr[:, idx] = torch.sum(fmap1 * shifted, dim=1)
            idx += 1
    
    dim = fmap1.shape[1]
    return corr / torch.sqrt(torch.tensor(dim).float())

class CPUCostVolume:
    """
    Custom CPU-safe wrapper that correctly implements the 4-level correlation pyramid
    expected by the SmallUpdateBlock (196 channels).
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=3): # Default to 4 levels, radius 3 for small model
        self.num_levels = num_levels
        self.radius = radius

        # Create the feature pyramid for fmap2
        self.pyramid = [fmap2]
        current_fmap2 = fmap2
        for i in range(self.num_levels - 1):
            # Downsample feature map for next level
            current_fmap2 = F.avg_pool2d(current_fmap2, 2, stride=2)
            self.pyramid.append(current_fmap2)
            
        self.fmap1 = fmap1

    def __call__(self, coords):
        
        # coords are ignored for the unwarped CPU-safe correlation, but are used
        # to determine the dimensions (B, H, W) of the GRU's input.
        coords = coords.permute(0, 2, 3, 1) # [B, 2, H/8, W/8] -> [B, H/8, W/8, 2]
        B, H, W, _ = coords.shape
        
        corr_list = []
        # The target resolution for all correlation outputs is the resolution of self.fmap1
        H_base, W_base = self.fmap1.shape[2], self.fmap1.shape[3]
        
        for i in range(self.num_levels):
            r = self.radius
            fmap2_i = self.pyramid[i] # fmap2_i is at resolution H/8, H/16, H/32, H/64...
            
            # Simplified CPU Approximation:
            # 1. We UP-SAMPLE the current pyramid level of fmap2 back to the base GRU resolution (H/8, W/8).
            fmap2_i_upsampled = F.interpolate(fmap2_i, (H_base, W_base), mode='nearest')
            
            # 2. We perform the localized correlation between the base fmap1 and the upsampled fmap2_i.
            # This ensures the output 'corr' is consistently sized at (B, 49, H_base, W_base).
            corr = localized_corr(self.fmap1, fmap2_i_upsampled, r)
            corr_list.append(corr)

        # Concatenate all 4 volumes (4 * 49 = 196 channels)
        # All tensors in corr_list have the same spatial dimensions (H_base, W_base),
        # so concatenation is safe.
        out = torch.cat(corr_list, dim=1)
        return out.contiguous().float()
