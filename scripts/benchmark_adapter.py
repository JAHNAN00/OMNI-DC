"""OMNI-DC v1.0: published NYU config, complete CG integration, random weights."""
from collections import Counter
import math
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODEL_NAME = 'OMNI-DC'
DCN_BACKEND = 'none (DySPN grid_sample + original conjugate-gradient integration)'


def model_config(seed, data_dir):
    previous = sys.argv
    try:
        sys.argv = ['config.py', '--gpus', '0', '--seed', str(seed),
                    '--num_resolution', '3', '--GRU_iters', '1',
                    '--optim_layer_input_clamp', '1.0', '--depth_activation_format', 'exp',
                    '--whiten_sparse_depths', '1', '--gru_internal_whiten_method', 'median',
                    '--backbone_mode', 'rgbd', '--pred_confidence_input', '1',
                    '--multi_resolution_learnable_gradients_weights', 'uniform',
                    '--load_dav2', '0', '--dir_data', str(data_dir)]
        args = runpy.run_path(str(ROOT / 'src/config.py'))['args']
    finally:
        sys.argv = previous
    args.from_scratch = True
    args.variant = 'v1.0 (testing_scripts/test_void_nyu.sh)'
    args.input_hw = [228, 304]
    args.internal_hw = [240, 304]
    args.padding_lrtb = [0, 0, 0, 12]
    args.cg_rtol = 1e-5
    args.cg_maxiter = 5000
    return args


def build_model(args):
    from model.ognidc import OGNIDC
    return OGNIDC(args)


class DepthOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('K', torch.tensor([
            [582.62448167737955 / 2, 0, 313.04475870804731 / 2 - 8],
            [0, 582.69103270988637 / 2, 238.44389626620386 / 2 - 6],
            [0, 0, 1]], dtype=torch.float32).unsqueeze(0), persistent=False)
        self.register_buffer('pattern', torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, rgb, dep):
        result = self.net({'rgb': F.pad(rgb, self.net.args.padding_lrtb),
                           'dep': F.pad(dep, self.net.args.padding_lrtb),
                           'K': self.K, 'pattern': self.pattern})['pred']
        return result[:, :, :dep.shape[-2], :dep.shape[-1]]


def reference_prediction(model, rgb, dep):
    sample = {'rgb': F.pad(rgb, model.net.args.padding_lrtb),
              'dep': F.pad(dep, model.net.args.padding_lrtb),
              'K': model.K, 'pattern': model.pattern}
    return model.net(sample)['pred'][:, :, :dep.shape[-2], :dep.shape[-1]]


def profile_macs(model, rgb, dep):
    """Count neural Conv/Linear/attention MACs, never claim to count CG arithmetic."""
    from optim_layer import optim_layer
    from pvt import Attention
    counts, handles, solver_calls = Counter(), [], []

    def count(module, inputs, result):
        if isinstance(module, nn.Conv2d):
            counts['conv'] += result.numel() * module.weight[0].numel()
        elif isinstance(module, nn.ConvTranspose2d):
            counts['conv'] += inputs[0].numel() * math.prod(module.weight.shape[1:])
        elif isinstance(module, nn.Linear):
            counts['linear'] += result.numel() * module.in_features
        elif isinstance(module, Attention):
            batch, tokens, channels = inputs[0].shape
            h, w = inputs[1:3]
            reduced = (h // module.sr_ratio) * (w // module.sr_ratio) if module.sr_ratio > 1 else tokens
            counts['matmul'] += 2 * batch * tokens * reduced * channels

    original_cg = optim_layer.cg_batch

    def observe_cg(*args, **kwargs):
        result, info = original_cg(*args, **kwargs)
        solver_calls.append(dict(info))
        return result, info

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, Attention)):
                handles.append(module.register_forward_hook(count))
        with patch.object(optim_layer, 'cg_batch', observe_cg), torch.inference_mode():
            model(rgb, dep)
    finally:
        for handle in handles:
            handle.remove()
    total = sum(counts.values())
    return {'macs': total, 'macs_g': total / 1e9, 'by_operator': dict(counts),
            'coverage': 'PARTIAL: neural-network Conv/Linear/attention only',
            'definition': 'nominal Conv (including transpose), Linear and attention matmul; excludes CG solver/finite differences/reductions, DySPN grid_sample, normalization and elementwise operations',
            'solver_calls_on_profile_sample': solver_calls,
            'note': 'CG data-dependent iterations and CPU/GPU convergence synchronization remain included in measured latency'}
