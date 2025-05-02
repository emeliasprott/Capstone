import datetime, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GINEConv, LEConv, RGATConv
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes

hidden_dim = 256
n_layers = 2
dropout_p = 0.15
device = torch.device('mps')

policy_topic_embs = torch.load('policy_embeddings.pt')
num_topics = len(policy_topic_embs)
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
            elif float(t) > 17000000.0:
                td = t
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
    timestamp_nodes = ['legislator_term', 'bill_version', 'bill', 'committee']

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

def load_and_preprocess_data(path='data2.pt'):
    full_data = torch.load(path, weights_only=False)
    data = ToUndirected(merge=True)(full_data)
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

class TimeEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        inv = 1. / 10 ** torch.linspace(0, 4, dim // 2)
        self.register_buffer("inv", inv)

    def forward(self, t: torch.Tensor):
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = torch.einsum("..., d -> ... d", t, self.inv)
        return torch.cat([freqs.sin(), freqs.cos()], -1)

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

class FeatureProjector(nn.Module):
    def __init__(self, metadata, hidden, in_dims, t_dim=16):
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

def calculate_modularity(edge_index, cluster_assignment, num_nodes):
    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).float()
    m = max(edge_index.size(1) // 2, 1)
    same = cluster_assignment[row] == cluster_assignment[col]
    exp = degree[row] * degree[col] / (2.0 * m)
    modularity = (same.float() - exp[same]).sum() / (2.0 * m)
    return modularity

class UnsupervisedTopicClustering(nn.Module):
    def __init__(self, in_channels, num_clusters=30, dropout_p=0.15):
        super().__init__()
        self.num_clusters = num_clusters

        self.assignment_net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_channels, num_clusters)
        )

    def forward(self, x, edge_index=None):
        s = self.assignment_net(x)
        s = F.softmax(s, dim=-1) + 1e-8
        s = s / (s.sum(dim=-1, keepdim=True) + 1e-10)
        device = x.device
        loss = torch.tensor(0.0, device=device, requires_grad=True)


        if edge_index is not None:
            try:
                hard_assignment = s.argmax(dim=-1)
                mod_loss = calculate_modularity(edge_index, hard_assignment, s.size(0))
                if not isinstance(mod_loss, torch.Tensor):
                    mod_loss = torch.tensor(float(mod_loss), device=device)
                loss = -mod_loss * (s.sum() / s.size(0))

            except Exception as e:
                print(f"Error in modularity calculation: {e}")
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        return s, loss
    def get_clusters(self, x):
        s = self.assignment_net(x)
        return s.argmax(dim=-1)

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
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
        return self.gine(x_to_use, edge_index, gated)

class TopicProjector(nn.Module):
    def __init__(self, hidden, num_topics):
        super().__init__()
        self.proj = nn.Linear(hidden, num_topics)
    def forward(self, z):
        return self.proj(z)

class AggregatedInfluenceScorer(nn.Module):
    def __init__(self, actor_dim, bill_dim):
        super().__init__()
        self.proj_actor = nn.Linear(actor_dim, actor_dim)
        self.proj_bill = nn.Linear(bill_dim, actor_dim)
        self.scorer = nn.Linear(actor_dim, 1)

    def forward(self, actor_embeddings, bill_embeddings, bill_outcomes):
        actor_proj = self.proj_actor(actor_embeddings)
        bill_proj = self.proj_bill(bill_embeddings)

        actor_proj = actor_proj.unsqueeze(1)
        bill_proj = bill_proj.unsqueeze(0)

        affinity = torch.tanh(actor_proj + bill_proj)
        scores = self.scorer(affinity).squeeze(-1)

        weighted_scores = scores * bill_outcomes.unsqueeze(0)

        influence_scores = weighted_scores.mean(dim=1)

        return influence_scores

class ContextualTopicAligner(nn.Module):
    def __init__(self, actor_dim, bill_dim, num_topics):
        super().__init__()
        self.proj_actor = nn.Linear(actor_dim, actor_dim)
        self.proj_bill = nn.Linear(bill_dim, actor_dim)
        self.topic_proj = nn.Linear(actor_dim, num_topics)

    def forward(self, actor_embeddings, bill_embeddings):
        actor_proj = self.proj_actor(actor_embeddings)
        bill_proj = self.proj_bill(bill_embeddings)

        actor_proj = actor_proj.unsqueeze(1)
        bill_proj = bill_proj.unsqueeze(0)

        fusion = torch.tanh(actor_proj + bill_proj)
        logits = self.topic_proj(fusion)

        return logits

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

        self.encoder = FeatureProjector(metadata=self.metadata, hidden=hidden_dim, in_dims=self.in_dims, t_dim=16)
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
        self.topic_clustering = UnsupervisedTopicClustering(
            in_channels=hidden_dim,
            num_clusters=int(num_topics),
            dropout_p=float(dropout_p)
        )
        self.influence_scorer = AggregatedInfluenceScorer(hidden_dim, hidden_dim)
        self.contextual_topic_head = ContextualTopicAligner(hidden_dim, hidden_dim, num_topics)

        self.policy_topic_matcher = PolicyTopicMatcher(hidden_dim, policy_topic_embs).to(self.device)

        self.controversy_head = nn.Linear(hidden_dim, 1)
        _init_linear(self.controversy_head)

        self.affinity_proj = nn.Linear(hidden_dim, hidden_dim)
        _init_linear(self.affinity_proj)

        self.denoise_proj = nn.ModuleDict({nt: nn.Linear(hidden_dim, data[nt].x.size(-1)) for nt in data.node_types})
        for lin in self.denoise_proj.values():
            _init_linear(lin)

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
                msg = self.polarity_conv((h_src, h_dst), edge_index, edge_attr)
                if msg.size(0) == h_dst.size(0):
                    x_dict[dst_type] = h_dst + msg
                else:
                    x_dict[dst_type] = h_dst.clone()
                    dst_nodes = edge_index[1]
                    x_dict[dst_type].index_add_(0, dst_nodes, msg)

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

        bill_emb = x_dict['bill']
        bill_outcomes = torch.from_numpy(np.vstack(batch['bill'].y)).to(self.device)

        emb_list = []
        offsets = {}
        cursor = 0
        for nt in batch.node_types:
            if nt in ['legislator', 'committee', 'donor', 'lobby_firm']:
                n = x_dict[nt].size(0)
                offsets[nt] = (cursor, cursor + n)
                emb_list.append(x_dict[nt])
                cursor += n
        flat_emb = torch.cat(emb_list, dim=0)
        flat_emb = torch.nan_to_num(flat_emb, nan=0.0, posinf=1e4, neginf=-1e4)

        flat_contextual_topic_scores = torch.nan_to_num(self.contextual_topic_head(flat_emb, bill_emb), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        flat_influence_scores = torch.nan_to_num(self.influence_scorer(flat_emb, bill_emb, bill_outcomes), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        flat_bill_topic_probs = torch.nan_to_num(self.policy_topic_matcher(bill_emb), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        controversy_pred = torch.nan_to_num(self.controversy_head(x_dict['bill_version']).squeeze(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        def get_offset_scores(offsets, scores):
            result = {}
            for nt in offsets.keys():
                s,e = offsets[nt]
                result[nt] = scores[s:e]
            return result
        influence_scores = get_offset_scores(offsets, flat_influence_scores)
        contextual_topic_scores = get_offset_scores(offsets, flat_contextual_topic_scores)
        bill_topic_probs = get_offset_scores(offsets, flat_bill_topic_probs)

        return {
            'node_embeddings': x_dict,
            'topic_scores': contextual_topic_scores,
            'flat_topic_scores': flat_contextual_topic_scores,
            'topic_dists': bill_topic_probs,
            'flat_topic_dists': flat_bill_topic_probs,
            'influence_scores': influence_scores,
            'flat_influence_scores': flat_influence_scores,
            'controversy_pred': controversy_pred
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
                    n_samples = min(valid_src_idx.size(0) * num_neg, 10000)
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

    def compute_loss(self, z_dict, data):
        z_dict = outputs['node_embeddings']

        try:
            link_loss = self.compute_link_reconstruction_loss(z_dict, data)
        except Exception as e:
            print(f"Error in link reconstruction: {e}")
            link_loss = torch.tensor(0.0, device=self.device)

        inf = torch.nan_to_num(outputs['flat_influence_scores'], nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        influence_reg_loss = (inf.clamp(-1e3, 1e3) ** 2).mean()


        topic_align_loss = torch.tensor(0.0, device=self.device)
        for scores in outputs['topic_scores'].values():
            probs = F.softmax(scores, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            topic_align_loss += entropy

        if "bill_version" in z_dict and hasattr(data["bill_version"], "controversy"):
            tgt = data['bill_version'].controversy.float().to(self.device)
            pred = outputs['controversy_pred']
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            if pred.numel() == tgt.numel():
                controversy_loss = F.mse_loss(pred, tgt, reduction='mean')
            else:
                controversy_loss = torch.tensor(0.0, device=self.device)
        else:
            controversy_loss = torch.tensor(0.0, device=self.device)

        topic_align_loss = topic_align_loss / (len(outputs['topic_scores']) + 1e-6)

        batch_size_factor = sum(x.size(0) for x in z_dict.values()) / sum(1 for _ in z_dict.values())
        batch_size_factor = max(1.0, batch_size_factor / 100)

        total_loss = (link_loss + (influence_reg_loss * 0.05) + (topic_align_loss * 0.05) + controversy_loss) / batch_size_factor

        return {
            'total_loss': total_loss,
            'link_loss': link_loss,
            'influence_reg_loss': influence_reg_loss,
            'topic_align_loss': topic_align_loss,
            'controversy_loss': controversy_loss
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


best_loss = float('inf')
best_embeddings = None
model = LegislativeGraphEncoder(num_topics, data.metadata(), hidden_dim, n_layers, dropout_p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
from torch_geometric.loader import RandomNodeLoader


best_loss = float('inf')
best_embeddings = None
model = LegislativeGraphEncoder(num_topics, data.metadata(), hidden_dim, n_layers, dropout_p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

best_loss = float('inf')
best_embeddings = None
model = LegislativeGraphEncoder(num_topics, data.metadata(), hidden_dim, n_layers, dropout_p).to(device)

state_dict = torch.load("best__model3.pt")
model.load_state_dict(state_dict)
