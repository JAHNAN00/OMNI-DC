import sys

import torch
import torch.utils.model_zoo

from scripts.benchmark_nyu import NYUInputs, ROOT, build_model, model_config, summary, DepthOnly


def test_random_initialization_never_loads_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Random inference must not load or download weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    monkeypatch.setattr(torch.utils.model_zoo, 'load_url', forbidden)
    torch.set_num_threads(1)
    args = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    net = build_model(args)
    assert args.from_scratch and not args.load_dav2
    assert net.GRU_iters == 1 and net.resolution == 3 and net.prop_time == 6
    assert not hasattr(net, 'depth_module')
    assert 'mmcv' not in sys.modules and 'mmseg' not in sys.modules and 'apex' not in sys.modules


def test_nyu_sampling_and_shape():
    dataset = NYUInputs(ROOT / 'data/nyudepthv2_h5')
    first, repeat = dataset[0], dataset[0]
    assert len(dataset) == 654
    assert first['rgb'].shape == (1, 3, 228, 304)
    assert first['dep'].shape == (1, 1, 228, 304)
    assert (first['dep'] > 0).sum() == 500
    assert torch.equal(first['dep'], repeat['dep'])


def test_released_nyu_configuration_and_padding_constraints():
    args = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    assert args.whiten_sparse_depths == 1 and args.pred_confidence_input == 1
    assert args.multi_resolution_learnable_gradients_weights == 'uniform'
    assert args.gru_internal_whiten_method == 'median' and args.depth_activation_format == 'exp'
    assert args.internal_hw == [240, 304]
    assert all(size % (args.backbone_output_downsample_rate * 2**(args.num_resolution - 1)) == 0
               for size in args.internal_hw)


def test_finite_difference_transpose_matches_adjoint():
    from optim_layer.helpers import FastFiniteDiffMatrix
    operator = FastFiniteDiffMatrix(8, 12, 3, device='cpu')
    x = torch.randn(1, 8 * 12, 1)
    ax = operator.bmm(x)
    y = torch.randn_like(ax)
    torch.testing.assert_close((ax * y).sum(), (x * operator.bmm_transposed(y)).sum(),
                               rtol=1e-5, atol=1e-5)


def test_summary():
    result = summary([10, 20, 30])
    assert result['mean_ms'] == 20 and result['p95_ms'] == 29 and result['fps'] == 50
