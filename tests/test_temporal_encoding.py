import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric.nn")
pytest.importorskip("torch_scatter")

import torch

import GNN.LeGNN as legnn


def test_temporal_encoder_responds_to_relative_features():
    torch.manual_seed(0)
    encoder = legnn.LegislativeTemporalEncoder(d=16)

    features = torch.zeros((2, 5), dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 0] = 1.0
    features[1, 1] = 0.5

    output = encoder(features, 'vote')

    assert output.shape == (2, 16)
    assert not torch.allclose(output[0], output[1])


def test_temporal_messages_use_relative_components(monkeypatch):
    class DummyHGTConv(torch.nn.Module):
        def __init__(self, in_dim, out_dim, metadata_subset, heads=1):
            super().__init__()
            self.metadata = metadata_subset
            self.out_dim = out_dim

        def forward(self, x_dict, edge_index_dict):
            return {nt: torch.zeros_like(x) for nt, x in x_dict.items()}

    monkeypatch.setattr(legnn, "HGTConv", DummyHGTConv)

    metadata = (
        ('donor', 'legislator_term'),
        [('donor', 'donated_to', 'legislator_term')],
    )

    encoder = legnn.LegislativeGraphEncoder(
        hidden_dim=4,
        dropout=0.0,
        metadata=metadata,
        relation_weight_sharing=None,
        device='cpu',
        heads=1,
    )

    class DummyTemporalEncoder(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, features, process_type):
            summed = features.sum(dim=-1, keepdim=True)
            return summed.repeat(1, self.hidden_dim)

    encoder.temporal_encoder = DummyTemporalEncoder(encoder.hidden_dim)

    x_dict = {
        'donor': torch.zeros((1, encoder.hidden_dim)),
        'legislator_term': torch.zeros((2, encoder.hidden_dim)),
    }

    edge_index = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    edge_index_dict = {('donor', 'donated_to', 'legislator_term'): edge_index}
    edge_attr_dict = {}

    edge_ts_dict = {('donor', 'donated_to', 'legislator_term'): torch.tensor([1.0, 2.0])}
    node_ts_dict = {
        'donor': torch.tensor([0.5]),
        'legislator_term': torch.tensor([1.5, 2.5]),
    }
    edge_delta_dict = {('donor', 'donated_to', 'legislator_term'): torch.tensor([0.1, 0.2])}

    out = encoder(
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        edge_ts_dict,
        node_ts_dict,
        edge_delta_dict,
    )

    expected = torch.tensor([3.1, 6.2], dtype=torch.float32).unsqueeze(-1)
    expected = expected.repeat(1, encoder.hidden_dim)

    assert torch.allclose(out['legislator_term'], expected, atol=1e-6)
    assert torch.allclose(out['donor'], x_dict['donor'])
    assert encoder.process_map[('donor', 'donated_to', 'legislator_term')] == 'donation'
    assert encoder.process_map['donated_to'] == 'donation'
