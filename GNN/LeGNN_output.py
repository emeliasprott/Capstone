import datetime, gc, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import SAGEConv, GINEConv, LEConv
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_geometric.loader import NeighborLoader

hidden_dim = 128
n_layers = 2
dropout_p = 0.15
device = torch.device('mps')
policy_topic_embs = torch.load('GNN/policy_embeddings.pt')
num_topics = len(policy_topic_embs)

# data preprocessing
def compute_controversiality(data):
    edge_type = ('legislator_term', 'voted_on', 'bill_version')
    if edge_type not in data.edge_index_dict:
        raise ValueError("Missing 'voted_on' edges in data.")

    ei = data[edge_type].edge_index
    ea = data[edge_type].edge_attr

    vote_signal = ea[:, 0]

    src_nodes = ei[0]
    tgt_nodes = ei[1]

    num_bills = data['bill_version'].num_nodes
    device = tgt_nodes.device

    yes_votes = torch.zeros(num_bills, device=device)
    no_votes = torch.zeros(num_bills, device=device)

    yes_votes.index_add_(0, tgt_nodes, (vote_signal > 0).float())
    no_votes.index_add_(0, tgt_nodes, (vote_signal < 0).float())

    total_votes = yes_votes + no_votes + 1e-6

    yes_ratio = yes_votes / total_votes
    no_ratio = no_votes / total_votes

    controversy = 4 * yes_ratio * no_ratio
    controversy = controversy.clamp(0, 1)
    data['bill_version'].controversy = controversy

    return data

def safe_normalize_timestamps(timestamps: torch.Tensor) -> torch.Tensor:
    timestamps = torch.nan_to_num(timestamps, nan=0.0, posinf=1e4, neginf=-1e4)
    min_time = timestamps.min()
    max_time = timestamps.max()
    if (max_time - min_time) < 1e-4:
        return torch.zeros_like(timestamps)
    return (timestamps - min_time) / (max_time - min_time)

def safe_standardize_time_format(time_data) -> torch.Tensor:
    times = []
    for t in time_data:
        try:
            if isinstance(t, (int, float)) and 1900 <= t  and t <= 2100:
                td = datetime.datetime(int(t), 6, 15).timestamp()
            elif (isinstance(t, str) or (isinstance(t, float))) and (float(t) < 2100 and float(t) > 1900):
                td = datetime.datetime(int(float(t)), 6, 15).timestamp()
            elif float(t) > 0 and float(t) < 1990:
                td = t
            elif float(t) > 17000000.0:
                td = float(t)
                while td > 10:
                    td = td / 10
            elif float(t) < 0:
                td = -float(t)
            else:
                td = t.timestamp()
        except:
            td = datetime.datetime(2000, 6, 15).timestamp()
        times.append(td)
    return torch.tensor(times, dtype=torch.float32)

def pull_timestamps(data):
    timestamp_edges = [
        ('donor', 'donated_to', 'legislator_term'),
        ('legislator_term', 'rev_donated_to', 'donor'),
        ('lobby_firm', 'lobbied', 'legislator_term'),
        ('lobby_firm', 'lobbied', 'committee'),
        ('committee', 'rev_lobbied', 'lobby_firm'),
        ('legislator_term', 'rev_lobbied', 'lobby_firm'),
        ('bill_version', 'rev_voted_on', 'legislator_term'),
        ('legislator_term', 'voted_on', 'bill_version'),
    ]
    timestamp_nodes = ['legislator_term', 'bill_version', 'bill']

    for et in timestamp_edges:
        if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None and len(data[et].edge_attr.size()) > 1:
            if data[et].edge_attr.size(1) > 1:
                edge_attr = data[et].edge_attr
                ts_col = edge_attr[:, -1]
                if ts_col.abs().max() > 1e8 or ts_col.min() < 0:
                    ts_col = safe_standardize_time_format(ts_col.tolist()).to(edge_attr.device)
                data[et].timestamp = safe_normalize_timestamps(ts_col)
                data[et].edge_attr = edge_attr[:, :-1]

    for nt in timestamp_nodes:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            try:
                if len(data[nt].x.size()) > 1:
                    if data[nt].x.size(1) > 1:
                        x = data[nt].x
                        ts_col = x[:, -1]
                        if ts_col.abs().max() > 1e8 or ts_col.min() < 0:
                            ts_col = safe_standardize_time_format(ts_col.tolist()).to(x.device)
                        if nt in timestamp_nodes or ts_col.abs().max() > 1e6:
                            data[nt].timestamp = safe_normalize_timestamps(ts_col)
                            data[nt].x = x[:, :-1]
            except:
                pass
    return data
def clean_features(data):
    for nt in data.node_types:
        x = data[nt].x
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            data[nt].x = torch.from_numpy(np.vstack(x)).float()
            x = data[nt].x
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if x.size(0) < 2 or torch.all(x == x[0]):
            mean = x.clone()
            std = torch.ones_like(x)
            x_clean = x.clone()
        else:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp(min=1e-5)
            x_clean = (x - mean) / std
            x_clean = x_clean.clamp(-10.0, 10.0)
        data[nt].x = x_clean
        data[nt].x_mean = mean
        data[nt].x_std = std
    data = pull_timestamps(data)
    return data

def load_and_preprocess_data(path='GNN/data2.pt'):
    full_data = torch.load(path, weights_only=False)
    for nt in full_data.node_types:
        if hasattr(full_data[nt], 'x') and full_data[nt].x is not None:
            full = torch.from_numpy(full_data[nt].x)
            s = full.size()
            full = torch.flatten(full, start_dim=1, end_dim=-1)
            full_data[nt].x = full
            full_data[nt].num_nodes = full.size(0)

    for edge_type, edge_index in full_data.edge_index_dict.items():
        src_type, _, dst_type = edge_type

        max_src_idx = edge_index[0].max().item() if edge_index.size(1) > 0 else -1
        max_dst_idx = edge_index[1].max().item() if edge_index.size(1) > 0 else -1

        if max_src_idx >= full_data[src_type].num_nodes:
            print(f"Fixing {src_type} node count: {full_data[src_type].num_nodes} -> {max_src_idx + 1}")
            full_data[src_type].num_nodes = max_src_idx + 1

        if max_dst_idx >= full_data[dst_type].num_nodes:
            print(f"Fixing {dst_type} node count: {full_data[dst_type].num_nodes} -> {max_dst_idx + 1}")
            full_data[dst_type].num_nodes = max_dst_idx + 1

    data = ToUndirected(merge=False)(full_data)
    del full_data
    gc.collect()
    data = RemoveIsolatedNodes()(data)
    data = compute_controversiality(clean_features(data))
    for store in data.stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                store[key] = value.float()
    return data

data = load_and_preprocess_data()

# utilities
def _init_linear(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=0.01)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def sanitize(t: torch.Tensor, clamp: float = 1e4) -> torch.Tensor:
    if t.dtype == torch.float64:
        t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
    return t.clamp_(-clamp, clamp)

def safe_mse(pred, tgt):
    return F.mse_loss(pred, tgt, reduction="mean")

def scaled_cosine(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T

# encoders
class TimeEncoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        inv = 1. / 8 ** torch.linspace(0, 4, dim // 2)
        self.register_buffer("inv", inv)

    def forward(self, t: torch.Tensor):
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = torch.einsum("..., d -> ... d", t, self.inv)
        return torch.cat([freqs.sin(), freqs.cos()], -1)

class FeatureProjector(nn.Module):
    def __init__(self, metadata, hidden, in_dims, t_dim=8):
        super().__init__()
        self.t_dim = t_dim
        self.time_encoder = TimeEncoder(t_dim)

        self.projectors = nn.ModuleDict()
        for nt in metadata[0]:
            use_time = nt in ['legislator_term', 'bill_version', 'committee', 'bill']
            in_dim = in_dims[nt]
            total_in = in_dim + (t_dim if use_time else 0)

            self.projectors[nt] = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden, bias=False),
                nn.GELU()
            )
            _init_linear(self.projectors[nt][1])

    def forward(self, x_dict, timestamp_dict=None):
        result = {}
        for nt, x in x_dict.items():
            x = sanitize(x)
            if timestamp_dict is not None and nt in timestamp_dict and timestamp_dict[nt] is not None:
                time_features = self.time_encoder(timestamp_dict[nt])
                x = torch.cat([x, time_features], dim=-1)
            result[nt] = self.projectors[nt](x)
        return result

# attention & convolution layers
class PolarityAwareConv(nn.Module):
    def __init__(self, hidden_dim, edge_attr_dim, dropout_p, in_channels, out_channels):
        super().__init__()
        feature_dim = edge_attr_dim - 1
        self.dropout_p = dropout_p
        self.edge_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.gine = GINEConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(out_channels, out_channels),
            ),
            edge_dim=hidden_dim
        )

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, tuple):
            x_src, x_dst = x
            x_to_use = x_src
        else:
            x_to_use = x
        polarity = edge_attr[:, 0:1].clamp(0,1)

        raw = edge_attr[:, 1:]
        if raw.dim() == 1:
            raw = raw.unsqueeze(-1)
        e = self.edge_mlp(raw)
        gated = e * (polarity + 0.01)
        max_idx = x_to_use.size(0) - 1
        if torch.any(edge_index[0] > max_idx) or torch.any(edge_index[0] < 0):
            valid_mask = (edge_index[0] <= max_idx) & (edge_index[0] >= 0)
            edge_index = edge_index[:, valid_mask]
            gated = gated[valid_mask]

            if edge_index.size(1) == 0:
                if isinstance(x, tuple):
                    return torch.zeros_like(x_dst)
                else:
                    return torch.zeros_like(x)

        try:
            return self.gine(x_to_use, edge_index, gated)
        except Exception as e:
            print(f"Error in GINEConv: {e}")
            if isinstance(x, tuple):
                return x_dst
            else:
                return x

class TopicClusterHead(nn.Module):
    def __init__(self, policy_topic_embs, hidden_dim):
        super().__init__()
        raw_dim = policy_topic_embs.size(-1)

        self.topic_embs = nn.Parameter(policy_topic_embs)
        self.proj = (
            nn.Identity() if raw_dim == hidden_dim
            else nn.Linear(raw_dim, hidden_dim, bias=False)
        )

    def forward(self, bill_z):
        topic_z = self.proj(self.topic_embs)
        topic_scores = scaled_cosine(bill_z, topic_z)
        return topic_scores

class InfluenceScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.score_vec = nn.Linear(dim, 1, bias=False)

    def forward(self, actor_z, bill_z, bill_y):
        bill_y = bill_y.flatten()
        q = self.query_proj(actor_z)
        k = self.key_proj(bill_z)
        sim = torch.tanh(q.unsqueeze(1) + k.unsqueeze(0))
        pair = self.score_vec(sim).squeeze(-1)
        inf = (pair * bill_y.unsqueeze(0)).mean(dim=1)
        return inf

class ContextualTopicAligner(nn.Module):
    def __init__(self, actor_dim, bill_dim, num_topics):
        super().__init__()
        self.proj_actor = nn.Linear(actor_dim, actor_dim, bias=False)
        self.proj_bill  = nn.Linear(bill_dim,  actor_dim, bias=False)
        self.topic_proj = nn.Linear(actor_dim, num_topics, bias=False)

    def forward(self, actor_embeddings, bill_embeddings):
        a = self.proj_actor(actor_embeddings).unsqueeze(1)
        b = self.proj_bill(bill_embeddings).unsqueeze(0)
        fusion = torch.tanh(a + b)
        return self.topic_proj(fusion)

class PolicyTopicMatcher(nn.Module):
    def __init__(self, bill_dim, topic_embs):
        super().__init__()
        self.policy_topic_embs = F.normalize(topic_embs, p=2, dim=-1)
        self.proj = nn.Linear(bill_dim, self.policy_topic_embs.size(-1))

    def forward(self, bill_embeddings):
        projected = F.normalize(self.proj(bill_embeddings), p=2, dim=-1)
        scores = torch.matmul(projected, self.policy_topic_embs.t())
        probs = F.softmax(scores, dim=-1)
        return probs

class ActorTopicAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, actor_z, bill_z, bill_topic_prob):
        attn_logits = scaled_cosine(self.query_proj(actor_z), self.key_proj(bill_z))
        alpha = attn_logits.softmax(dim=1)
        return alpha @ bill_topic_prob

class LegislativeGraphEncoder(nn.Module):
    def __init__(self, num_topics, metadata, hidden_dim, num_layers, dropout, device=device):
        super().__init__()
        self.device = device
        self.metadata = metadata
        self.node_types = ['bill', 'bill_version', 'legislator_term', 'committee', 'donor', 'lobby_firm', 'legislator']
        self.edge_types = [
            ('legislator_term', 'voted_on', 'bill_version'),
            ('donor', 'donated_to', 'legislator_term'),
            ('lobby_firm', 'lobbied', 'legislator_term'),
            ('legislator_term', 'member_of', 'committee'),
            ('bill_version', 'is_version', 'bill'),
            ('bill_version', 'priorVersion', 'bill_version'),
            ('legislator', 'samePerson', 'legislator_term'),
            ('lobby_firm', 'lobbied', 'committee'),
            ('legislator_term', 'wrote', 'bill_version'),
        ]
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        self.num_layers = num_layers
        self.dropout = dropout

        self.in_dims = {nt: data[nt].x.shape[-1] for nt in data.node_types}

        self.encoder = FeatureProjector(metadata=self.metadata, hidden=hidden_dim, in_dims=self.in_dims, t_dim=8)
        self.sage_conv = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        vote_edge_type = ('legislator_term','voted_on','bill_version')
        vote_attr_dim  = data[vote_edge_type].edge_attr.size(-1)
        self.polarity_conv = PolarityAwareConv(
            hidden_dim=hidden_dim,
            edge_attr_dim=vote_attr_dim,
            dropout_p=dropout_p,
            in_channels=hidden_dim,
            out_channels=hidden_dim
        )

        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim).to(self.device) for node_type in data.node_types
        })
        self.le_conv = LEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.final_linear = nn.Linear(hidden_dim, hidden_dim).to(self.device)

    def process_edges(self, x_dict, data):
        for edge_type, edge_index in data.edge_index_dict.items():
            edge_index = edge_index.to(self.device)
            src_type, rel, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            h_src = x_dict[src_type]
            h_dst = x_dict[dst_type]
            if edge_type == ('legislator_term', 'voted_on', 'bill_version'):
                edge_attr = data[edge_type].edge_attr.to(self.device)
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]

                # Make sure we're only using valid indices
                valid_src = src_nodes < h_src.size(0)
                valid_dst = dst_nodes < h_dst.size(0)
                valid_mask = valid_src & valid_dst

                if not valid_mask.any():
                    continue

                # Filter edge_index and edge_attr
                filtered_edge_index = edge_index[:, valid_mask]
                filtered_edge_attr = edge_attr[valid_mask]
                filtered_dst_nodes = filtered_edge_index[1]

                try:
                    msg = self.polarity_conv((h_src, h_dst), filtered_edge_index, filtered_edge_attr)

                    # Create a new destination embedding tensor
                    new_dst = h_dst.clone()

                    # Get unique destination nodes that received messages
                    unique_dst_nodes = torch.unique(filtered_dst_nodes)

                    # Only update nodes that received messages
                    if msg.size(0) == unique_dst_nodes.size(0):
                        # If we have one message per unique destination node
                        new_dst[unique_dst_nodes] = new_dst[unique_dst_nodes] + msg
                    elif msg.size(0) == filtered_dst_nodes.size(0):
                        # If we have one message per edge, aggregate them
                        for i, node_idx in enumerate(unique_dst_nodes):
                            # Find all messages for this node
                            node_mask = (filtered_dst_nodes == node_idx)
                            # Aggregate messages (e.g., by taking the mean)
                            node_msgs = msg[node_mask]
                            if node_msgs.size(0) > 0:
                                new_dst[node_idx] = new_dst[node_idx] + node_msgs.mean(dim=0)

                    # Update the node embeddings
                    x_dict[dst_type] = new_dst

                except Exception as e:
                    print(f"Error in polarity_conv: {e}")
                    # Keep original embeddings if there's an error
                    x_dict[dst_type] = h_dst

            elif edge_type in [
                    ('donor', 'donated_to', 'legislator_term'),
                    ('lobby_firm', 'lobbied', 'legislator_term'),
                    ('legislator_term', 'member_of', 'committee')
                ]:
                msg = self.le_conv((h_src, h_dst), edge_index)

                if msg.size(0) == h_dst.size(0):
                    x_dict[dst_type] = h_dst + msg
                else:
                    x_dict[dst_type] = h_dst.clone()
                    dst_nodes = edge_index[1]
                    x_dict[dst_type].index_add_(0, dst_nodes, msg)
            else:
                with torch.no_grad():
                    msg = self.sage_conv((h_src, h_dst), edge_index)
                if msg.size(0) == h_dst.size(0):
                    x_dict[dst_type] = 0.8 * h_dst + 0.2 * msg
                else:
                    x_dict[dst_type] = 0.9 * h_dst.clone()
                    dst_nodes = edge_index[1]
                    x_dict[dst_type].index_add_(0, dst_nodes, 0.1 * msg)

        return x_dict

    def add_attention(self, x_dict, edge_type, edge_index):
        src_type, rel, dst_type = edge_type
        if src_type not in x_dict or dst_type not in x_dict:
            return x_dict
        h_src = x_dict[src_type]
        h_dst = x_dict[dst_type]
        e_attr = data[edge_type].edge_attr.to(self.device)
        edge_dim = e_attr.size(-1)
        msg = self.attention((h_src, h_dst), edge_index, e_attr)
        x_dict[dst_type] = h_dst + msg
        return x_dict

    def apply_normalization(self, x_dict):
        for nt in x_dict:
            if nt in self.norm:
                x_dict[nt] = self.norm[nt](x_dict[nt])
                x_dict[nt] = F.relu(x_dict[nt])
                x_dict[nt] = self.dropout(x_dict[nt])
                x_dict[nt] = self.final_linear(x_dict[nt])
        return x_dict

    def encode_nodes(self, data):
        timestamp_dict = {}
        x_dict = {}
        for nt in data.node_types:
            x_dict[nt] = data[nt].x.to(self.device)
            if hasattr(data[nt], "timestamp"):
                timestamp_dict[nt] = data[nt].timestamp.to(self.device)
            else:
                timestamp_dict[nt] = None
        h_dict = {nt: torch.zeros(data[nt].num_nodes, self.hidden_dim, device=self.device) for nt in data.node_types}

        encoded = self.encoder(x_dict, timestamp_dict=timestamp_dict)
        for nt, h in encoded.items():
            if nt in h_dict:
                h_dict[nt] = h
        return h_dict
    def forward(self, batch=None):
        if batch is None:
            batch = self.data
        x_dict = self.encode_nodes(batch)
        x_dict = self.process_edges(x_dict, batch)
        x_dict = self.apply_normalization(x_dict)

        return x_dict

class LegislativeGraphModel(nn.Module):
    def __init__(self, metadata, hidden_dim, policy_topic_embs, device=device):
        super().__init__()
        self.device = device
        self.encoder = LegislativeGraphEncoder(num_topics, metadata, hidden_dim, n_layers, dropout_p, device)
        self.topic_cluster = TopicClusterHead(policy_topic_embs, hidden_dim)
        self.actor_types   = ["legislator", "committee", "donor", "lobby_firm"]
        self.influence = nn.ModuleDict({
            nt: InfluenceScorer(hidden_dim) for nt in self.actor_types
        })
        self.topic_align = nn.ModuleDict({
            nt: ActorTopicAligner(hidden_dim) for nt in self.actor_types
        })

    def forward(self, data):
        z = self.encoder(data)
        bill_z = z['bill']
        bill_y = torch.from_numpy(np.vstack(data['bill'].y)).float().to(self.device)
        bill_topic_logits = self.topic_cluster(bill_z)
        bill_topic_prob = bill_topic_logits.softmax(dim=-1)

        influence_dict, actor_topic_dict = {}, {}
        for nt in self.actor_types:
            if nt in z:
                actor_z = z[nt]
                influence_dict[nt] = self.influence[nt](actor_z, bill_z, bill_y)
                actor_topic_dict[nt] = self.topic_align[nt](actor_z, bill_z, bill_topic_prob)

        return {
            'node_embeddings': z,
            'bill_topic_logits': bill_topic_logits,
            'bill_topic_prob': bill_topic_prob,
            'influence_dict': influence_dict,
            'actor_topic_dict': actor_topic_dict
        }

    def compute_link_reconstruction_loss(self, z_dict, data, num_neg=1):
        loss = torch.tensor(0., device=self.device)

        for rel, ei in data.edge_index_dict.items():
            s_t, t, d_t = rel
            if s_t not in z_dict or d_t not in z_dict:
                continue
            src, dst = ei
            valid_src = (src < z_dict[s_t].size(0))
            valid_dst = (dst < z_dict[d_t].size(0))
            valid_mask = valid_src & valid_dst
            if not valid_mask.any():
                continue
            valid_src_idx = src[valid_mask]
            valid_dst_idx = dst[valid_mask]
            src_emb = F.normalize(z_dict[s_t][valid_src_idx], p=2, dim=-1)
            dst_emb = F.normalize(z_dict[d_t][valid_dst_idx], p=2, dim=-1)
            pos_score = torch.clamp((src_emb * dst_emb).sum(dim=-1), -10, 10)
            pos_loss = nn.BCEWithLogitsLoss()(pos_score, torch.ones_like(pos_score, device=self.device))
            try:
                max_dst_idx = z_dict[d_t].size(0) - 1
                if max_dst_idx > 0:
                    n_samples = min(valid_src_idx.size(0) * num_neg, 5000)
                    neg_dst = torch.randint(0, max_dst_idx + 1, (n_samples,), device=self.device)
                    neg_src_indices = torch.randint(0, valid_src_idx.size(0), (n_samples,), device=self.device)
                    neg_src = valid_src_idx[neg_src_indices]
                    neg_src_emb = F.normalize(z_dict[s_t][neg_src], p=2, dim=-1)
                    neg_dst_emb = F.normalize(z_dict[d_t][neg_dst], p=2, dim=-1)
                    neg_score = torch.clamp((neg_src_emb * neg_dst_emb).sum(dim=-1), -10, 10)
                    neg_loss = nn.BCEWithLogitsLoss()(neg_score, torch.zeros_like(neg_score, device=self.device))
                    loss += pos_loss + neg_loss
            except Exception as e:
                print(f"Error in negative sampling: {e}")
                loss += pos_loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf detected in final link reconstruction loss")
            return torch.tensor(0.01, device=self.device, requires_grad=True)
        return loss

    def compute_loss(self, data):
        out = self.forward(data)
        z = out['node_embeddings']

        link_loss = self.compute_link_reconstruction_loss(z, data)
        all_inf = torch.cat(list(out['influence_dict'].values()))
        inf_reg = (all_inf ** 2).mean()

        ent = 0.0
        for probs in out['actor_topic_dict'].values():
            p = probs.clamp_min(1e-9)
            ent += (-p * p.log()).sum(dim=-1).mean()
        ent_reg = ent / max(len(out['actor_topic_dict']), 1)

        total = link_loss + (0.1 * inf_reg) + (0.1 * ent_reg)
        return {
            'total_loss': total,
            'link_loss': link_loss,
            'inf_reg': inf_reg,
            'ent_reg': ent_reg
        }


    def get_topic_clusters(self, batch=None):
        if batch is None:
            batch = self.data
        outputs = self.forward(batch)
        z_dict = outputs['node_embeddings']
        clusters = {}
        if 'bill' in z_dict:
            clusters['bill'] = self.topic_clustering.get_clusters(z_dict['bill'])

        return clusters

    def get_influence_scores(self, batch=None):
        if batch is None:
            batch = self.data

        outputs = self.forward(batch)
        return outputs['influence_scores']

    def get_attention_heads(self, batch=None):
        if batch is None:
            batch = self.data

        outputs = self.forward(batch)
        z_dict = outputs['node_embeddings']

        attention_heads = {}

        if 'topic_dists' in outputs and 'bill' in outputs['topic_dists']:
            attention_heads['topic_attention'] = outputs['topic_dists']['bill']

        if 'influence_scores' in outputs:
            for nt, scores in outputs['influence_scores'].items():
                attention_heads[f'{nt}_influence'] = scores
        if 'alignment_loss' in outputs:
            attention_heads['alignment_loss'] = outputs['alignment_loss']

        return attention_heads

def bill_kpis(bill_ids, p_topic, actor_topic, influence):
    dom = p_topic.argmax(1)
    entr = (-p_topic * p_topic.clamp(min=1e-9).log()).sum(1)

    cos_sim = F.cosine_similarity(
        p_topic.unsqueeze(1),
        actor_topic.unsqueeze(0),
        dim=-1
    )
    inf = influence.view(-1)
    total_inf = inf.sum()

    align_sup = (cos_sim * inf).sum(1) / total_inf

    mean = align_sup.unsqueeze(1)
    var = ((cos_sim - mean)**2 * inf).sum(1) / total_inf
    polar = var.sqrt()

    risk = torch.sigmoid(3 * align_sup + 1 * inf.mean() - 2 * polar)

    return pd.DataFrame({
        "bill_id": bill_ids.cpu().numpy(),
        "dominant_topic": dom.cpu().numpy(),
        "topic_entropy": entr.cpu().numpy(),
        "alignment_support": align_sup.cpu().numpy(),
        "polarisation_score": polar.cpu().numpy(),
        "success_risk": risk.cpu().numpy(),
    })


def actor_kpis(node_ids, p_topic, influence, ideology_index=(0,1)):
    assert len(node_ids) == p_topic.size(0)
    focus = (p_topic**2).sum(1)
    lever = focus * influence.view(-1)

    if p_topic.size(1) > max(ideology_index):
        bipartisan = 1 - torch.abs(
            p_topic[:, ideology_index[0]] - p_topic[:, ideology_index[1]]
        )
    else:
        bipartisan = torch.ones_like(lever)

    return pd.DataFrame({
        "node_id": node_ids.cpu().numpy(),
        "top_topic": p_topic.argmax(1).cpu().numpy(),
        "topic_focus": focus.cpu().numpy(),
        "influence": influence.view(-1).cpu().numpy(),
        "leverage": lever.cpu().numpy(),
        "bipartisan_score": bipartisan.cpu().numpy(),
    })


def topic_snapshot(p_topic_bills, bill_dates, actor_topic_all, influence_all):
    df = pd.DataFrame({
        "date": pd.to_datetime(bill_dates),
        "topic": p_topic_bills.argmax(1).cpu().numpy()
    })
    weekly = df.groupby(["topic", pd.Grouper(key="date", freq="W")]).size()

    momentum = {}
    for topic, group in weekly.groupby(level=0):
        counts = group.values
        if len(counts) < 2:
            slope = 0.0
        else:
            try:
                slope = np.polyfit(np.arange(len(counts)), counts, 1)[0]
            except Exception:
                slope = 0.0
        momentum[topic] = slope

    power = torch.matmul(actor_topic_all.t(), influence_all.view(-1))
    num_topics = p_topic_bills.size(1)
    if power.numel() >= 2:
        diff = float((power[0] - power[1]).item())
        balance_vals = np.full(num_topics, diff, dtype=float)
    else:
        balance_vals = power.cpu().numpy().astype(float)

    return pd.DataFrame({
        "topic_id": np.arange(num_topics),
        "recent_momentum": [momentum.get(t, 0.0) for t in range(num_topics)],
        "power_balance": balance_vals
    })

torch.mps.empty_cache()
gc.collect()
for nt in data.node_types:
    data[nt].node_id = torch.arange(data[nt].num_nodes, dtype=torch.long)

model = LegislativeGraphModel(data.metadata(), hidden_dim, policy_topic_embs).to(device)
state_dict = torch.load("best_model5.pt")
model.load_state_dict(state_dict)

loader = NeighborLoader(
    data,
    num_neighbors={k: [25] * 2 for k in data.edge_types},
    batch_size=128,
    shuffle=False,
    input_nodes=('legislator_term')
)

embs = {nt: [] for nt in data.node_types}
bill_probs_parts = []
influence_parts = {nt: [] for nt in data.node_types}
actor_topic_parts = {nt: [] for nt in data.node_types}
OUT_DIR = Path("dashboard/backend/data")

shared_paths = {tbl: [] for tbl in [
    "bill_kpis", "topic_snapshot",
    *[f"{nt}_kpis" for nt in data.node_types],
    *[f"{nt}_embeddings" for nt in data.node_types]
]}

for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Processing batches"):
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch)
    z_dict = out['node_embeddings']
    bill_prob = out['bill_topic_prob']
    infl_dict = out['influence_dict']
    actor_topic = out['actor_topic_dict']

    bill_ids = batch['bill'].node_id.cpu()
    p_topic_bills = bill_prob.index_select(0, bill_ids.to(bill_prob.device))
    timestamp_full = data['bill'].timestamp
    bill_dates = timestamp_full[bill_ids].cpu().numpy()
    actor_topic_all = torch.cat([actor_topic[nt] for nt in actor_topic], 0)
    influence_all = torch.cat([infl_dict[nt].view(-1) for nt in infl_dict], 0)
    bill_df = bill_kpis(
        bill_ids,
        p_topic_bills.cpu(),
        actor_topic_all.cpu(),
        influence_all.cpu()
    )
    bill_df.to_parquet(OUT_DIR / f"bill_kpis_{i}.parquet", index=False)
    shared_paths['bill_kpis'].append(f"{OUT_DIR}/bill_kpis_{i}.parquet")

    for nt in infl_dict:
        node_ids = batch[nt].node_id
        df = actor_kpis(
            node_ids.cpu(),
            actor_topic[nt].cpu(),
            infl_dict[nt].cpu()
        )
        p = OUT_DIR / f"{nt}_kpis_{i}.parquet"
        df.to_parquet(p, index=False)
        shared_paths[f"{nt}_kpis"].append(str(p))
        pd.DataFrame({
            "node_id": node_ids.cpu().numpy(),
            "topic_probs": list(actor_topic[nt].cpu().numpy())
        }).to_parquet(OUT_DIR / f"{nt}_topic_probs_{i}.parquet", index=False)


        e = OUT_DIR / f"{nt}_embeddings_{i}.parquet"
        emb_df = pd.DataFrame({
            "node_id": node_ids.cpu().numpy(),
            "embedding": list(z_dict[nt].cpu().numpy())
        })
        emb_df.to_parquet(e, index=False)
        shared_paths[f"{nt}_embeddings"].append(str(e))
    snap_df = topic_snapshot(
        p_topic_bills.cpu(),
        bill_dates,
        actor_topic_all.cpu(),
        influence_all.cpu()
    )
    s = OUT_DIR / f"topic_snapshot_{i}.parquet"
    snap_df.to_parquet(s, index=False)
    shared_paths['topic_snapshot'].append(str(s))

with open(OUT_DIR / "shared_paths.json", "w") as f:
    json.dump(shared_paths, f)
