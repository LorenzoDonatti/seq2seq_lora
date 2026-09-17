import json
import numpy as np
import pytest
import torch
from src.data_loader import create_sliding_windows
from src.models.dlinear import NLinear, DLinear
from src.models.multi_node_seq2seq import MultiNodeSeq2SeqAttention
from src.models.training import fit_network
from src.config_store import load_optimized_config


def test_future_weather_is_not_required_but_future_targets_are():
    data = np.ones((4, 2), dtype=np.float32)
    data[-1, 1] = np.nan
    X, y, origins = create_sliding_windows(data, 3, 1, 1, return_origins=True)
    assert len(X) == 1 and origins.tolist() == [3]
    data[-1, 0] = np.nan
    assert len(create_sliding_windows(data, 3, 1, 1)[0]) == 0


@pytest.mark.parametrize("cls", [NLinear, DLinear])
def test_original_linear_is_channel_independent_and_ignores_weather(cls):
    torch.manual_seed(1)
    model = cls(24, 6, 12, 8)
    x = torch.randn(3, 24, 12)
    altered = x.clone()
    altered[:, :, 1:] += 100
    torch.testing.assert_close(model(x)[:, :, 0], model(altered)[:, :, 0])
    expected = 25 * 6 * (2 if cls is DLinear else 1)
    assert sum(p.numel() for p in model.parameters()) == expected


def test_nlinear_matches_original_shared_formula():
    model = NLinear(24, 6, 12, 8)
    x = torch.randn(3, 24, 12)
    rssi = x[:, :, :8]
    expected = model.linear((rssi-rssi[:, -1:]).transpose(1,2)).transpose(1,2)+rssi[:, -1:]
    torch.testing.assert_close(model(x), expected)


def test_dlinear_matches_original_decomposition():
    model = DLinear(24, 6, 12, 8)
    x = torch.randn(3, 24, 12)
    rssi = x[:, :, :8].transpose(1, 2)
    trend = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(rssi, (12,12), mode="replicate"), 25, stride=1)
    expected = (model.seasonal(rssi-trend)+model.trend(trend)).transpose(1,2)
    torch.testing.assert_close(model(x), expected)


def test_seq2seq_runs_recurrent_decoder_each_step_and_backpropagates():
    torch.manual_seed(2)
    model = MultiNodeSeq2SeqAttention(12,8,6,hidden_dim=8,dropout=0)
    inputs = []
    hook = model.decoder.register_forward_pre_hook(lambda m,a: inputs.append(a[0]))
    x = torch.randn(2,24,12,requires_grad=True)
    prediction = model(x)
    hook.remove()
    assert prediction.shape == (2,6,8) and len(inputs) == 6
    torch.testing.assert_close(inputs[0][:,:8], x[:,-1,:8])
    torch.testing.assert_close(inputs[1][:,:8], prediction[:,0])
    prediction[:,-1].sum().backward()
    assert model.decoder.weight_hh.grad.abs().sum() > 0
    assert model.attention.v.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_incompatible_configuration_cache_is_rejected(tmp_path):
    path=tmp_path/"cache.json"
    path.write_text(json.dumps({"experiments": {"missing:missing.csv": {
        "histories": {"24": {"configs": {"ARIMAX": {}}}}
    }}}))
    with pytest.raises(ValueError,match="optimize"):
        load_optimized_config(data_file="missing.csv",history=24,path=path)


def test_checkpoint_restores_best_validation_and_uses_physical_units():
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): model.weight.fill_(0)
    x=np.ones((4,1),dtype=np.float32)
    summary=fit_network(model,"cpu",x,x,x,-x,lr=.1,weight_decay=0,
                        epochs=10,batch_size=4,target_scale=np.array([3.]),patience=2)
    assert summary["best_epoch"]==1 and summary["epochs_executed"]==3
    actual=float(((model(torch.from_numpy(x))+1).abs()*3).mean().item())
    assert actual==pytest.approx(summary["best_validation_loss"])
