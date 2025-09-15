import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric.data")

import torch
from torch_geometric.data import HeteroData

from GNN.LeGNN import compute_controversiality


def build_vote_data():
    data = HeteroData()
    data['legislator_term'].num_nodes = 3
    data['bill_version'].num_nodes = 2
    data['bill_version'].session = torch.tensor([2021, 2022])
    data['bill_version'].total_possible_votes = torch.tensor([5.0, 4.0])

    edge_index = torch.tensor([
        [0, 1, 2, 0, 1],
        [0, 0, 0, 1, 1],
    ])
    vote_signal = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0]).unsqueeze(1)

    data['legislator_term', 'voted_on', 'bill_version'].edge_index = edge_index
    data['legislator_term', 'voted_on', 'bill_version'].edge_attr = vote_signal
    return data


def test_compute_controversiality_accounts_for_abstentions():
    data = HeteroData()
    data['legislator_term'].num_nodes = 3
    data['bill_version'].num_nodes = 1

    edge_index = torch.tensor([
        [0, 1, 2],
        [0, 0, 0],
    ])
    vote_signal = torch.tensor([1.0, -1.0, 0.0]).unsqueeze(1)

    data['legislator_term', 'voted_on', 'bill_version'].edge_index = edge_index
    data['legislator_term', 'voted_on', 'bill_version'].edge_attr = vote_signal

    result = compute_controversiality(data)

    assert torch.allclose(result['bill_version'].abstain_votes, torch.tensor([1.0]))
    assert torch.allclose(result['bill_version'].total_possible_votes, torch.tensor([3.0]))

    expected = torch.tensor([8.0 / 27.0])
    assert torch.allclose(result['bill_version'].controversy, expected, atol=1e-6)


def test_compute_controversiality_uses_total_possible_and_session():
    data = build_vote_data()

    result = compute_controversiality(
        data,
        session_attr='session',
        total_possible_attr='total_possible_votes',
    )

    expected = torch.tensor([0.064, 0.125])
    assert torch.allclose(result['bill_version'].controversy, expected, atol=1e-6)

    session_stats = result['bill_version'].session_controversy
    assert session_stats[2021] == pytest.approx(0.064)
    assert session_stats[2022] == pytest.approx(0.125)
