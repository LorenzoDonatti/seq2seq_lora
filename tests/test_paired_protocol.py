import numpy as np
import pytest
import torch
from src.models.baselines import BoxJenkinsARIMAX, HistoricalWeatherVARX
from src.models.stgnn import AdaptiveSTGNN, AdaptiveGraphConvolution, SpatioTemporalBlock
from src.node_topology import compute_physical_adjacency
from src.model_registry import DEFAULTS, NEURAL_MODELS, GRAPH_ABLATIONS, make_neural, model_metadata
from src.runtime import seed_everything
from src.statistical_search import _arimax_candidate_orders


def test_box_jenkins_shortlist_is_bounded_and_uses_admissible_difference():
    diagnostic = {
        "admissible_differences": [0],
        "by_difference": {"0": {"suggested_p": 4, "suggested_q": 3}},
    }
    orders = _arimax_candidate_orders(diagnostic)
    assert 1 <= len(orders) <= 8
    assert all(order[1] == 0 for order in orders)
    assert (4, 0, 3) in orders


def test_varx_forecast_uses_only_delayed_observed_weather():
    model = HistoricalWeatherVARX(2, 3, 1)
    model.coefficients = [np.array([0.,0.,0.,2.]),np.array([0.,0.,0.,3.])]
    X = np.zeros((1,8,3))
    X[:,-3:,2] = [1.,2.,3.]
    expected = np.array([[[2.,3.],[4.,6.],[6.,9.]]])
    np.testing.assert_allclose(model.predict(X),expected)
    changed = X.copy()
    changed[:,-1,2] = 100
    np.testing.assert_allclose(model.predict(changed)[:,:2],expected[:,:2])
    assert not np.allclose(model.predict(changed)[:,-1],expected[:,-1])


def test_sparse_var_lags_use_only_declared_history_positions():
    model = HistoricalWeatherVARX(1, 1, [1, 3])
    model.coefficients = [np.array([0., 2., 5., 0.])]
    X = np.zeros((1, 3, 2))
    X[0, :, 0] = [7., 11., 13.]
    # 2 * lag-1 + 5 * lag-3; lag 2 is deliberately absent.
    np.testing.assert_allclose(model.predict(X)[0, 0, 0], 2 * 13 + 5 * 7)
    assert model.lag_indices == [1, 3]


def test_varx_fit_recovers_known_weather_effect_and_resets_gaps():
    rng=np.random.default_rng(44)
    segments=[]
    for _ in range(2):
        x=rng.normal(size=250)
        y=np.zeros((250,2))
        for t in range(3,250):
            y[t]=.5*y[t-1]+np.array([2.,3.])*x[t-3]
        segments.append(np.column_stack([y,x]))
    model=HistoricalWeatherVARX(2,3,1,alpha=0)
    model.fit(np.zeros((1,12,3)),np.zeros((1,3,2)),training_segments=segments)
    assert model.training_summary['training_segment_lengths']==[250,250]
    assert model.training_summary['conditional_observations']==494
    for node,beta in enumerate(model.coefficients):
        assert beta[-1]==pytest.approx(2+node,abs=1e-6)


def test_box_jenkins_arimax_uses_node_specific_orders_and_delayed_weather():
    rng = np.random.default_rng(12)
    weather = rng.normal(size=(100, 1))
    target = np.zeros(100)
    for t in range(1, 100):
        target[t] = 0.4 * target[t-1] + 0.7 * weather[t-1, 0] + rng.normal(scale=.05)
    segment = np.column_stack([target, weather])
    model = BoxJenkinsARIMAX(1, 1, [(1, 0, 0)])
    X = np.stack([segment[70:82], segment[80:92]]).astype(np.float32)
    model.fit(X, np.zeros((2, 1, 1)), training_segments=[segment[:70]])
    prediction = model.predict(X)
    assert prediction.shape == (2, 1, 1)
    assert np.isfinite(prediction).all()
    assert model.training_summary["orders"] == [[1, 0, 0]]


@pytest.mark.parametrize("mode", ["none","physical","adaptive","hybrid"])
def test_graph_modes_have_no_self_edges_and_finite_gradients(mode):
    seed_everything(3)
    model=AdaptiveSTGNN(n_targets=3,n_exogenous=2,seq_length=12,pred_length=6,
                        hidden_dim=8,num_blocks=2,dropout=0,graph_mode=mode)
    x=torch.randn(2,12,5,requires_grad=True)
    prediction=model(x)
    assert prediction.shape==(2,6,3)
    prediction.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for block in model.blocks:
        adj=block.spatial_gcn.get_adjacency()
        torch.testing.assert_close(adj.diag(),torch.zeros(3))
        expected=torch.zeros(3) if mode=='none' else torch.ones(3)
        torch.testing.assert_close(adj.sum(-1),expected)
        if mode in ('adaptive','hybrid'):
            assert block.spatial_gcn.source_embedding.grad.abs().sum()>0


def test_graph_without_edges_has_no_cross_node_dependency():
    model=AdaptiveSTGNN(n_targets=3,n_exogenous=2,seq_length=12,pred_length=3,
                        hidden_dim=8,dropout=0,graph_mode='none').eval()
    x=torch.randn(2,12,5)
    changed=x.clone()
    changed[:,:,1:3]+=100
    torch.testing.assert_close(model(x)[:,:,0],model(changed)[:,:,0])


def test_graph_block_is_causal_even_with_dilation():
    block=SpatioTemporalBlock(3,8,compute_physical_adjacency(node_indices=[0,1,2]),
                             dilation=4,dropout=0).eval()
    x=torch.randn(2,16,3,8)
    changed=x.clone()
    changed[:,8:]+=10
    torch.testing.assert_close(block(x)[:,:8],block(changed)[:,:8])


def test_graph_edge_orientation_is_receiver_sender():
    graph=AdaptiveGraphConvolution(2,2,3,np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=np.float32),
                                   graph_mode='physical')
    with torch.no_grad():
        graph.w_spatial.weight.copy_(torch.eye(2))
        graph.w_self.weight.zero_(); graph.w_self.bias.zero_()
    x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])
    expected=graph.layer_norm(torch.relu(x[:,[1,2,0]]))
    torch.testing.assert_close(graph(x),expected)


@pytest.mark.parametrize('name', NEURAL_MODELS)
def test_paired_neural_trainers_fit_and_keep_local_inputs_local(name):
    seed_everything(4)
    rng=np.random.default_rng(4)
    X=rng.normal(size=(8,12,7)).astype(np.float32)
    y=rng.normal(size=(8,3,3)).astype(np.float32)
    data={'train':(X,y),'target_names':['a','b','c']}
    cfg={**DEFAULTS[name],'hidden_dim':8,'blocks':1,'dropout':0}
    model=make_neural(name,cfg,data,3,'cpu')
    model.fit(X[:6],y[:6],X[6:],y[6:],epochs=1,batch_size=6,
              target_scale=np.ones(3))
    prediction=model.predict(X)
    assert prediction.shape==y.shape and np.isfinite(prediction).all()
    if not model_metadata(name,3)['cross_node_inputs']:
        changed=X.copy(); changed[:,:,1:3]+=100
        np.testing.assert_allclose(model.predict(changed)[:,:,0],prediction[:,:,0],rtol=1e-5,atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('horizon',[1,24])
def test_cuda_all_neural_families_and_graph_ablations(horizon):
    # Tiny synthetic validation only: never selects hyperparameters on the test set.
    seed_everything(9)
    rng=np.random.default_rng(9)
    X=rng.normal(size=(8,48,12)).astype(np.float32)
    y=rng.normal(size=(8,horizon,8)).astype(np.float32)
    data={'train':(X,y),'target_names':[str(i) for i in range(8)]}
    for name in list(NEURAL_MODELS)+list(GRAPH_ABLATIONS):
        cfg={**DEFAULTS.get(name,DEFAULTS['PhysicalAdaptive_STGNN']),
             'hidden_dim':8,'blocks':1,'dropout':0}
        model=make_neural(name,cfg,data,horizon,'cuda')
        model.fit(X[:6],y[:6],X[6:],y[6:],epochs=1,batch_size=6,target_scale=np.ones(8))
        prediction=model.predict(X[6:])
        assert prediction.shape==(2,horizon,8) and np.isfinite(prediction).all()
