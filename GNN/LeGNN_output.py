import datetime, gc, json, torch, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_scatter import scatter_mean
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv
from contextlib import contextmanager
from pathlib import Path
from GNN.time_utils import convert_to_utc_seconds as _convert_to_utc_seconds_list
from collections.abc import Mapping

# utilities
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

_EDGE_SEP = '∷'


def _et_to_key(et):
    return _EDGE_SEP.join(et)


def _key_to_et(k):
    return tuple(k.split(_EDGE_SEP))


def _looks_like_edge(value):
    if isinstance(value, str):
        return value.count(_EDGE_SEP) == 2
    if isinstance(value, (list, tuple)):
        return len(value) == 3
    return False


def _normalize_edge_type(edge_type):
    if isinstance(edge_type, str):
        return _key_to_et(edge_type)
    edge_tuple = tuple(edge_type)
    if len(edge_tuple) != 3:
        raise ValueError(f"Edge type {edge_type!r} is not a valid (src, rel, dst) triplet")
    return edge_tuple


def _group_edge_types(edge_types, relation_weight_sharing=None):
    normalized = [tuple(et) for et in edge_types]
    edge_set = set(normalized)

    if relation_weight_sharing is None:
        groups = {}
        for src, rel, dst in normalized:
            group_key = rel[4:] if rel.startswith('rev_') else rel
            groups.setdefault(group_key, []).append((src, rel, dst))
        return {str(k): v for k, v in groups.items()}

    if isinstance(relation_weight_sharing, str):
        mode = relation_weight_sharing.lower()
        if mode in {"distinct", "separate", "none"}:
            return {str(_et_to_key(et)): [et] for et in normalized}
        if mode in {"all", "shared", "share_all"}:
            return {"all_relations": normalized}
        raise ValueError(f"Unknown relation_weight_sharing mode: {relation_weight_sharing}")

    if not isinstance(relation_weight_sharing, Mapping):
        raise TypeError("relation_weight_sharing must be a mapping, string, or None")

    groups = {}
    assigned = set()
    key_iterable = list(relation_weight_sharing.keys())
    is_edge_to_group = bool(key_iterable) and all(_looks_like_edge(key) for key in key_iterable)

    if is_edge_to_group:
        for raw_edge, group_name in relation_weight_sharing.items():
            edge = _normalize_edge_type(raw_edge)
            if edge not in edge_set:
                continue
            groups.setdefault(str(group_name), []).append(edge)
            assigned.add(edge)
    else:
        for group_name, relations in relation_weight_sharing.items():
            group_edges = []
            for rel in relations:
                edge = _normalize_edge_type(rel)
                if edge not in edge_set:
                    continue
                if edge in assigned:
                    raise ValueError(f"Edge type {edge} assigned to multiple groups")
                group_edges.append(edge)
                assigned.add(edge)
            if group_edges:
                groups[str(group_name)] = group_edges

    for edge in normalized:
        if edge not in assigned:
            groups.setdefault(_et_to_key(edge), []).append(edge)

    return groups

@contextmanager
def error_context(operation_name: str, logger: logging.Logger):
    try:
        logger.info(f"Starting {operation_name}")
        yield
        logger.info(f"Completed {operation_name}")
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        raise

def safe_tensor_operation(func, *args, default_value=0.0, clamp_range=(-1e3, 1e3), **kwargs):
    try:
        result = func(*args, **kwargs)
        if torch.isnan(result).any() or torch.isinf(result).any():
            logging.warning(f"NaN/Inf detected in {func.__name__}, replacing with {default_value}")
            result = torch.full_like(result, default_value)
        return torch.clamp(result, *clamp_range)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        return torch.tensor(default_value)

hidden_dim = 192
n_layers = 3
dropout_p = 0.10
device = 'mps'
inf_reg_weight = 0.1
ent_reg_weight = 0.1

DEFAULT_RELATION_WEIGHT_SHARING = {
    "donation": [
        ('donor', 'donated_to', 'legislator_term'),
        ('legislator_term', 'rev_donated_to', 'donor'),
    ],
    "lobbying": [
        ('lobby_firm', 'lobbied', 'legislator_term'),
        ('lobby_firm', 'lobbied', 'committee'),
        ('committee', 'rev_lobbied', 'lobby_firm'),
        ('legislator_term', 'rev_lobbied', 'lobby_firm'),
    ],
    "bill_hierarchy": [
        ('bill_version', 'is_version', 'bill'),
        ('bill', 'rev_is_version', 'bill_version'),
    ],
    "voting": [
        ('legislator_term', 'voted_on', 'bill_version'),
        ('bill_version', 'rev_voted_on', 'legislator_term'),
    ],
    "authorship": [
        ('legislator_term', 'wrote', 'bill_version'),
        ('bill_version', 'rev_wrote', 'legislator_term'),
    ],
    "membership": [
        ('committee', 'member_of', 'legislator_term'),
        ('committee', 'rev_member_of', 'legislator_term'),
        ('legislator_term', 'member_of', 'committee'),
    ],
}

def _init_linear(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=0.01)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def sanitize(t, clamp=1e4):
    t = t.float() if t.dtype == torch.float64 else t
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
    return t.clamp_(-clamp, clamp)


def _sanitize_feature_tensor(x):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        tensor = x.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(x, dtype=torch.float32)

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)

    return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

def _global_to_local(sorted_global, query):
    pos = torch.searchsorted(sorted_global, query)
    return pos

def alarm():
    import sounddevice as sd
    import time

    fs = 25500
    duration = 0.6
    frequency = 435

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = 1.05 * np.sin(2 * np.pi * frequency * t)

    for j in range(3):
        for i in range(3):
            sd.play(x, fs)
            sd.wait()

        time.sleep(0.5)

# preprocessing
def convert_to_utc_seconds(time_data):
    """Convert timestamp-like values to a torch tensor of UTC seconds."""

    return torch.tensor(_convert_to_utc_seconds_list(time_data), dtype=torch.float32)

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
                ts_seconds = convert_to_utc_seconds(ts_col).to(edge_attr.device)
                data[et].timestamp = ts_seconds
                data[et].edge_attr = edge_attr[:, :-1]

    for nt in timestamp_nodes:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            try:
                if len(data[nt].x.size()) > 1:
                    if data[nt].x.size(1) > 1:
                        x = data[nt].x
                        ts_col = x[:, -1]
                        ts_seconds = convert_to_utc_seconds(ts_col).to(x.device)
                        if nt in timestamp_nodes or ts_seconds.abs().max() > 1e6:
                            data[nt].timestamp = ts_seconds
                            data[nt].x = x[:, :-1]
            except:
                pass
    return data


def ensure_global_feature_stats(data):
    existing = getattr(data, 'feature_stats', None)
    stats = {}

    if isinstance(existing, dict):
        for nt, value in existing.items():
            if not isinstance(value, dict):
                continue
            mean = value.get('mean')
            std = value.get('std')
            if mean is None or std is None:
                continue
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32).detach().cpu()
            std_tensor = torch.as_tensor(std, dtype=torch.float32).detach().cpu().clamp(min=1e-5)
            stats[nt] = {'mean': mean_tensor, 'std': std_tensor}

    for nt in data.node_types:
        if nt in stats:
            continue

        store = data[nt]

        x_mean = getattr(store, 'x_mean', None)
        x_std = getattr(store, 'x_std', None)

        if x_mean is not None and x_std is not None:
            mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32).detach().cpu()
            std_tensor = torch.as_tensor(x_std, dtype=torch.float32).detach().cpu().clamp(min=1e-5)
            stats[nt] = {'mean': mean_tensor, 'std': std_tensor}
            continue

        x = getattr(store, 'x', None)
        sanitized = _sanitize_feature_tensor(x)
        if sanitized is None or sanitized.numel() == 0:
            continue

        mean_tensor = sanitized.mean(0, keepdim=True).detach().cpu()
        std_tensor = sanitized.std(0, keepdim=True).clamp(min=1e-5).detach().cpu()
        stats[nt] = {'mean': mean_tensor, 'std': std_tensor}

    data.feature_stats = stats
    return stats
def clean_features(data):
    stats = ensure_global_feature_stats(data)

    for nt in data.node_types:
        store = data[nt]
        x = getattr(store, 'x', None)
        sanitized = _sanitize_feature_tensor(x)
        if sanitized is None or sanitized.numel() == 0:
            continue

        node_stats = stats.get(nt)
        if node_stats is None:
            mean_tensor = sanitized.mean(0, keepdim=True).detach().cpu()
            std_tensor = sanitized.std(0, keepdim=True).clamp(min=1e-5).detach().cpu()
            node_stats = {'mean': mean_tensor, 'std': std_tensor}
            stats[nt] = node_stats

        mean = node_stats['mean'].to(device=sanitized.device, dtype=sanitized.dtype)
        std = node_stats['std'].to(device=sanitized.device, dtype=sanitized.dtype).clamp(min=1e-5)

        normalized = ((sanitized - mean) / std).clamp(-10, 10)

        store.x = normalized
        store.x_mean = mean
        store.x_std = std

    data.feature_stats = stats
    data = pull_timestamps(data)
    return data

def compute_controversiality(
    data,
    *,
    session_attr=None,
    total_possible_attr=None,
):
    """Compute controversy scores for bill versions based on vote signals."""

    edge_type = ('legislator_term', 'voted_on', 'bill_version')
    if edge_type not in data.edge_index_dict:
        raise ValueError("Missing 'voted_on' edges in data.")

    ei = data[edge_type].edge_index
    ea = data[edge_type].edge_attr

    vote_signal = ea[:, 0].to(torch.float32)

    tgt_nodes = ei[1]

    num_bills = data['bill_version'].num_nodes
    device = vote_signal.device

    def _to_tensor(value, *, length=None):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value)
        elif isinstance(value, (int, float)):
            tensor = torch.tensor([float(value)])
        else:
            return None

        tensor = tensor.to(device=device, dtype=torch.float32)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if length is not None:
            if tensor.numel() == 1 and length > 1:
                tensor = tensor.repeat(length)
            elif tensor.numel() != length:
                return None
        return tensor.view(-1)

    def _fetch_store_attr(store, name):
        if not isinstance(name, str):
            return None
        value = getattr(store, name, None)
        if value is None:
            try:
                value = store[name]
            except (KeyError, AttributeError, TypeError):
                value = None
        return value

    def _normalize_session_key(key):
        if isinstance(key, (int, float)):
            return int(key)
        if isinstance(key, str):
            candidate = key.strip()
            if candidate.isdigit():
                return int(candidate)
            try:
                return int(float(candidate))
            except ValueError:
                return None
        return None

    session_tensor = None
    if session_attr is not None:
        session_values = session_attr
        if isinstance(session_attr, str):
            session_values = _fetch_store_attr(data['bill_version'], session_attr)
        session_tensor = _to_tensor(session_values, length=num_bills)
        if session_tensor is None:
            raise ValueError(
                f"Session attribute '{session_attr}' could not be resolved to a tensor."
            )
        if torch.is_floating_point(session_tensor):
            session_tensor = session_tensor.round().long()
        else:
            session_tensor = session_tensor.to(torch.long)

    yes_votes = torch.zeros(num_bills, dtype=torch.float32, device=device)
    no_votes = torch.zeros(num_bills, dtype=torch.float32, device=device)
    abstain_votes = torch.zeros(num_bills, dtype=torch.float32, device=device)

    yes_votes.index_add_(0, tgt_nodes, (vote_signal > 0).float())
    no_votes.index_add_(0, tgt_nodes, (vote_signal < 0).float())
    abstain_votes.index_add_(0, tgt_nodes, (vote_signal == 0).float())

    observed_total = yes_votes + no_votes + abstain_votes

    candidate_attrs = []
    if total_possible_attr is not None:
        candidate_attrs.append(total_possible_attr)
    candidate_attrs.extend(
        [
            'total_possible_votes',
            'possible_votes',
            'committee_size',
            'chamber_size',
            'membership_size',
        ]
    )

    total_possible = None
    for attr in candidate_attrs:
        value = None
        if isinstance(attr, dict):
            if session_tensor is None:
                raise ValueError(
                    "Session information is required when total_possible_attr is a mapping."
                )
            mapped = torch.zeros(num_bills, dtype=torch.float32, device=device)
            for key, item in attr.items():
                normalized_key = _normalize_session_key(key)
                if normalized_key is None:
                    continue
                mask = session_tensor == normalized_key
                if not torch.any(mask):
                    continue
                fill_tensor = _to_tensor(item, length=None)
                if fill_tensor is None:
                    continue
                if fill_tensor.numel() == 1:
                    mapped[mask] = fill_tensor.item()
                elif fill_tensor.numel() == mask.sum().item():
                    mapped[mask] = fill_tensor.to(device=device, dtype=torch.float32)
                else:
                    raise ValueError(
                        "Mapping for total_possible_attr must provide either a scalar or "
                        "a tensor matching the number of bills in the session."
                    )
            value = mapped
        elif isinstance(attr, (str, torch.Tensor, np.ndarray, list, tuple, int, float)):
            if isinstance(attr, str):
                value = _fetch_store_attr(data['bill_version'], attr)
            else:
                value = attr
        if value is None:
            continue
        total_possible = _to_tensor(value, length=num_bills)
        if total_possible is not None:
            break

    if total_possible is None:
        total_possible = observed_total.clone()

    total_possible = torch.maximum(total_possible, observed_total)
    total_possible = total_possible.clamp(min=1.0)

    yes_ratio = yes_votes / total_possible
    no_ratio = no_votes / total_possible
    abstain_ratio = abstain_votes / total_possible
    participation_ratio = (yes_votes + no_votes) / total_possible

    controversy = 4 * yes_ratio * no_ratio * participation_ratio
    controversy = controversy.clamp(0, 1)

    data['bill_version'].controversy = controversy
    data['bill_version'].yes_votes = yes_votes
    data['bill_version'].no_votes = no_votes
    data['bill_version'].abstain_votes = abstain_votes
    data['bill_version'].total_possible_votes = total_possible
    data['bill_version'].participation_ratio = participation_ratio
    data['bill_version'].abstain_ratio = abstain_ratio

    if session_tensor is not None:
        unique_sessions, inverse = torch.unique(session_tensor, return_inverse=True)
        session_yes = torch.zeros(unique_sessions.size(0), dtype=torch.float32, device=device)
        session_no = torch.zeros_like(session_yes)
        session_abstain = torch.zeros_like(session_yes)
        session_total = torch.zeros_like(session_yes)

        session_yes.index_add_(0, inverse, yes_votes)
        session_no.index_add_(0, inverse, no_votes)
        session_abstain.index_add_(0, inverse, abstain_votes)
        session_total.index_add_(0, inverse, total_possible)

        session_total = session_total.clamp(min=1.0)
        session_yes_ratio = session_yes / session_total
        session_no_ratio = session_no / session_total
        session_participation = (session_yes + session_no) / session_total

        session_controversy = 4 * session_yes_ratio * session_no_ratio * session_participation
        session_controversy = session_controversy.clamp(0, 1)

        def _session_key(value):
            scalar = value.item()
            if isinstance(scalar, float) and scalar.is_integer():
                return int(scalar)
            return scalar

        data['bill_version'].session_controversy = {
            _session_key(sess.cpu()): float(session_controversy[i].item())
            for i, sess in enumerate(unique_sessions)
        }
        data['bill_version'].session_participation = {
            _session_key(sess.cpu()): float(session_participation[i].item())
            for i, sess in enumerate(unique_sessions)
        }

    return data

def load_and_preprocess_data(path='data3.pt', controversy_kwargs=None):
    full_data = torch.load(path, weights_only=False)
    for nt in full_data.node_types:
        if hasattr(full_data[nt], 'x') and full_data[nt].x is not None:
            flat = torch.as_tensor(full_data[nt].x).flatten(start_dim=1)
            full_data[nt].x = flat
            full_data[nt].num_nodes = flat.size(0)

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
    full_data['bill'].y[np.where(full_data['bill'].y < 0)[0]] = 0
    full_data['bill'].y = torch.as_tensor(full_data['bill'].y, dtype=torch.float32)

    data = ToUndirected(merge=False)(full_data)
    del full_data
    gc.collect()
    data = RemoveIsolatedNodes()(data)
    kwargs = controversy_kwargs or {}
    data = compute_controversiality(clean_features(data), **kwargs)

    for nt in data.node_types:
        ids = torch.arange(data[nt].num_nodes, device='mps')
        data[nt].node_id = ids
    for store in data.stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                store[key] = value.float()


    return data

# encoders
class LegislativeTemporalEncoder(nn.Module):
    def __init__(self, d=hidden_dim):
        super().__init__()
        self.hidden_dim = d
        for h in ['vote_temporal', 'donation_temporal', 'lobbying_temporal']:
            setattr(self, h, nn.Sequential(
                nn.Linear(1, d // 4),
                nn.ReLU(),
                nn.Linear(d // 4, d),
            ))

        self.vote_temporal = getattr(self, 'vote_temporal')
        self.donation_temporal = getattr(self, 'donation_temporal')
        self.lobbying_temporal = getattr(self, 'lobbying_temporal')

    def forward(self, timestamps, process_type):
        if process_type == 'vote':
            return self.vote_temporal(timestamps.unsqueeze(-1))
        elif process_type == 'donation':
            return self.donation_temporal(timestamps.unsqueeze(-1))
        elif process_type == 'lobbying':
            return self.lobbying_temporal(timestamps.unsqueeze(-1))
        else:
            return self.vote_temporal(timestamps.unsqueeze(-1))

class FeatureProjector(nn.Module):
    def __init__(self, in_dims, d_out=hidden_dim):
        super().__init__()
        self.prj = nn.Sequential(
            nn.LayerNorm(in_dims),
            nn.Linear(in_dims, d_out, bias=False),
            nn.GELU())

    def forward(self, x):
        return self.prj(x)

class PolarityAwareConv(nn.Module):
    def __init__(self, h_dim, e_dim, p):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(e_dim - 1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p),
        )

    def forward(self, ea):
        pol = ea[:, :1].clamp(0, 1)
        raw = ea[:, 1:]
        if raw.dim() == 1:
            raw = raw.unsqueeze(-1)

        e_feat = self.edge_mlp(raw) * (pol + 0.01)
        return e_feat

class ActorHead(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.attn = nn.MultiheadAttention(d, h, batch_first=True, dropout=0.0)

    def forward(self, a_z, bv_z, mask, weight=None):
        a_z = torch.nan_to_num(a_z, nan=0.0, posinf=1.0, neginf=-1.0)
        bv_z = torch.nan_to_num(bv_z, nan=0.0, posinf=1.0, neginf=-1.0)

        n_actor = a_z.size(0)
        device = a_z.device

        if n_actor == 0:
            topic_align = torch.zeros((0, self.dk), device=device, dtype=a_z.dtype)
            influence = torch.zeros(0, device=device, dtype=a_z.dtype)
            return topic_align, influence

        if mask.numel() == 0 or bv_z.numel() == 0:
            topic_align = torch.zeros((n_actor, self.dk), device=device, dtype=a_z.dtype)
            influence = torch.zeros(n_actor, device=device, dtype=a_z.dtype)
            return topic_align, influence

        mask = mask.to(device=device, dtype=torch.bool)
        valid_actor_mask = mask.any(dim=1)

        topic_align_full = torch.zeros((n_actor, self.dk), device=device, dtype=a_z.dtype)
        influence_full = torch.zeros(n_actor, device=device, dtype=a_z.dtype)

        if not valid_actor_mask.any():
            return topic_align_full, influence_full

        valid_idx = valid_actor_mask.nonzero(as_tuple=False).squeeze(-1)

        query = a_z[valid_idx].unsqueeze(1)
        key = bv_z.unsqueeze(0).expand(query.size(0), -1, -1)
        value = key

        mask_valid = mask[valid_idx]
        key_padding_mask = ~mask_valid

        attn_mask = None
        weight_valid = None
        if weight is not None and weight.numel() > 0:
            weight_valid = torch.nan_to_num(weight[valid_idx], nan=0.0, posinf=1.0, neginf=-1.0)
            weight_valid = weight_valid.masked_fill(key_padding_mask, float('-inf'))
            attn_mask = weight_valid.unsqueeze(1).repeat_interleave(self.h, dim=0).to(dtype=a_z.dtype)

        attn_output, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_output = torch.nan_to_num(attn_output.squeeze(1), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        topic_align_valid = attn_output.view(query.size(0), self.h, self.dk).mean(dim=1)

        attn_weights = torch.nan_to_num(attn_weights.squeeze(1), nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = attn_weights * mask_valid.float()

        if weight_valid is not None:
            norm_scores = F.softmax(weight_valid, dim=-1)
            norm_scores = torch.nan_to_num(norm_scores, nan=0.0, posinf=0.0, neginf=0.0)
            norm_scores = norm_scores * mask_valid.float()
            influence_valid = (norm_scores * attn_weights).sum(dim=-1)
        else:
            influence_valid = attn_weights.sum(dim=-1)

        influence_valid = torch.nan_to_num(influence_valid, nan=0.0, posinf=1.0, neginf=-1.0)

        topic_align_full[valid_idx] = topic_align_valid
        influence_full[valid_idx] = influence_valid

        return topic_align_full, influence_full

class BillTopicHead(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, k, bias=False)
    def forward(self, z):
        return self.fc(z)

class SuccessHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.l.weight, gain=0.1)
        nn.init.constant_(self.l.bias, 0.0)

    def forward(self, z):
        z = torch.clamp(z, -10, 10)
        logits = self.l(z).squeeze(-1)
        return torch.clamp(logits, -10, 10)


class LegislativeGraphEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout, metadata, relation_weight_sharing=None, device=device, heads=2):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        node_types, edge_types = metadata
        self.metadata = (tuple(node_types), [tuple(et) for et in edge_types])
        self.relation_groups = _group_edge_types(self.metadata[1], relation_weight_sharing)
        self.heads = heads

        self.process_map = {
            ('donor', 'donated_to', 'legislator_term'): 'donation',
            ('lobby_firm', 'lobbied', 'legislator_term'): 'lobbying',
            ('bill_version', 'is_version', 'bill'): 'hierarchy'
        }

        self.convs = nn.ModuleDict()
        for group_name, relations in self.relation_groups.items():
            metadata_subset = (self.metadata[0], relations)
            self.convs[group_name] = HGTConv(hidden_dim, hidden_dim, metadata_subset, heads=self.heads)

        self.temporal_encoder = LegislativeTemporalEncoder(hidden_dim)
        self.vote_conv = PolarityAwareConv(hidden_dim, 385, dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, ts_dict):
        out = {nt: torch.zeros_like(x, device=x.device) for nt, x in x_dict.items()}
        temporal_adjustments = {}

        if ts_dict is not None:
            for et, timestamps in ts_dict.items():
                if timestamps is None:
                    continue
                norm_et = _normalize_edge_type(et)
                if norm_et not in edge_index_dict:
                    continue
                edge_index = edge_index_dict[norm_et]
                if edge_index.numel() == 0:
                    continue
                src = norm_et[0]
                if src not in x_dict:
                    continue
                if edge_index[0].max() >= x_dict[src].size(0):
                    continue
                process = self.process_map.get(norm_et, 'vote')
                edge_temp = self.temporal_encoder(timestamps, process)
                node_temp = scatter_mean(edge_temp, edge_index[0], dim=0, dim_size=x_dict[src].size(0))
                temporal_adjustments[norm_et] = torch.nan_to_num(node_temp, nan=0.0, posinf=0.0, neginf=0.0)

        for group_name, conv in self.convs.items():
            relations = self.relation_groups[group_name]
            group_edge_index = {}
            src_modifiers = {}

            for et in relations:
                if et not in edge_index_dict:
                    continue
                edge_index = edge_index_dict[et]
                if edge_index.numel() == 0:
                    continue
                group_edge_index[et] = edge_index
                if et in temporal_adjustments:
                    src_modifiers.setdefault(et[0], []).append(temporal_adjustments[et])

            if not group_edge_index:
                continue

            x_group = x_dict
            if src_modifiers:
                x_group = x_dict.copy()
                for src_type, modifiers in src_modifiers.items():
                    stacked = torch.stack(modifiers)
                    delta = torch.nan_to_num(stacked.sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
                    x_group[src_type] = x_group[src_type] + delta

            conv_out = conv(x_group, group_edge_index)
            for nt, value in conv_out.items():
                if nt in out:
                    out[nt] += value
                else:
                    out[nt] = value

        vote_et = ('legislator_term', 'voted_on', 'bill_version')
        if vote_et in edge_attr_dict and vote_et in edge_index_dict:
            vote_attr = edge_attr_dict[vote_et]
            if vote_attr.numel() > 0:
                vote_edges = self.vote_conv(vote_attr)
                dst = edge_index_dict[vote_et][1]

                if dst.max() < out['bill_version'].size(0):
                    vote_msg_agg = scatter_mean(
                        vote_edges, dst, dim=0, dim_size=out['bill_version'].size(0)
                    )
                    out['bill_version'] += vote_msg_agg

        for nt in out:
            if nt in x_dict:
                out[nt] += x_dict[nt]

        return out

class LegislativeGraphModel(nn.Module):
    def __init__(self, in_dims, cluster_id, topic_onehot, hidden_dim, dropout, metadata, relation_weight_sharing=None, device=device):
        super().__init__()
        self.device = device

        self.node_types = in_dims.keys()
        self.metadata = (tuple(metadata[0]), [tuple(et) for et in metadata[1]])
        self.relation_weight_sharing = relation_weight_sharing

        self.feature_proj = nn.ModuleDict({
            nt: nn.Sequential(
                    nn.LayerNorm(in_dims[nt]),
                    nn.Linear(in_dims[nt], hidden_dim, bias=False),
                    nn.GELU(),
                )
            for nt in in_dims
        })
        self.cluster_id = cluster_id

        self.encoders = nn.ModuleList([
            LegislativeGraphEncoder(
                hidden_dim,
                dropout,
                self.metadata,
                relation_weight_sharing=self.relation_weight_sharing,
                device=device,
            )
            for _ in range(n_layers)
        ])

        self.bill_alpha = nn.Parameter(torch.tensor(0.7))
        self.leg_alpha  = nn.Parameter(torch.tensor(0.7))

        self.topic_head = BillTopicHead(hidden_dim, topic_onehot.size(1))
        self.success = SuccessHead(hidden_dim)
        self.topic_onehot = topic_onehot

        self.actor_types = ['legislator', 'committee', 'donor', 'lobby_firm']
        self.actor_head = nn.ModuleDict({nt: ActorHead(hidden_dim) for nt in self.actor_types})

        k_topics = topic_onehot.size(1)
        dk = hidden_dim // self.actor_head['legislator'].h
        self.actor_topic_proj = nn.ModuleDict({
            nt: nn.Linear(dk, k_topics, bias=False) for nt in self.actor_types
        })

        self.loss_computer = StableLossComputer(device=device)

    def _aggregate_hierarchy(self, z, b):
        s, d = b.edge_index_dict[('bill_version', 'is_version', 'bill')]
        bill_agg = scatter_mean(z['bill_version'][s], d,
                                dim=0, dim_size=z['bill'].size(0))
        z['bill'] = self.bill_alpha * z['bill'] + (1 - self.bill_alpha) * bill_agg

        d2, s2 = b.edge_index_dict[('legislator', 'samePerson', 'legislator_term')]
        leg_agg = scatter_mean(z['legislator_term'][s2], d2,
                               dim=0, dim_size=z['legislator'].size(0))
        z['legislator'] = self.leg_alpha * z['legislator'] + (1 - self.leg_alpha) * leg_agg
        return z

    def encoder(self, batch):
        x_dict = {
            nt: self.feature_proj[nt](batch[nt].x)
            for nt in self.feature_proj if hasattr(batch[nt], 'x')
        }

        ts_dict = {et: batch[et].timestamp for et in batch.edge_types
                   if hasattr(batch[et], 'timestamp')}

        for encoder in self.encoders:
            x_dict = encoder(x_dict, batch.edge_index_dict, batch.edge_attr_dict, ts_dict)

        return x_dict

    def forward(self, batch, mask_weight_dict):
        z = self._aggregate_hierarchy(self.encoder(batch), batch)
        bill_nodes = batch['bill'].node_id.to(self.device)
        bv_nodes = batch['bill_version'].node_id.to(self.device)

        bill_embed = z['bill'][bill_nodes]
        z_bv = z['bill_version'][bv_nodes]

        bill_logits = self.topic_head(bill_embed)
        bill_labels = self.cluster_id[bill_nodes]

        success_log = self.success(bill_embed)
        succ_scalar = torch.sigmoid(success_log).mean()

        align_dict, infl_dict = {}, {}
        for nt in self.actor_types:
            if nt not in z:
                continue
            mvb, wvb = mask_weight_dict[nt]
            ta, inf  = self.actor_head[nt](z[nt], z_bv, mvb.to(self.device), wvb.to(self.device))
            align_dict[nt] = self.actor_topic_proj[nt](ta).softmax(-1)
            infl_dict[nt]  = inf * succ_scalar

        return {
            "node_embeddings": z,
            "bill_logits": bill_logits,
            "bill_labels": bill_labels,
            "success_logit": success_log,
            "actor_topic_dict": align_dict,
            "influence_dict" : infl_dict
        }
    def compute_loss(self, outputs, batch):
        loss_dict = {}

        bill_logits = outputs["bill_logits"]
        bill_labels = outputs["bill_labels"]
        topic_loss = self.loss_computer.compute_topic_loss(bill_logits, bill_labels)
        loss_dict['topic_loss'] = topic_loss

        embeddings = outputs["node_embeddings"]
        link_loss = self.loss_computer.compute_link_loss(embeddings, batch.edge_index_dict)
        loss_dict['link_loss'] = link_loss

        success_loss = torch.tensor(0.0, device=self.device)
        if 'success_logit' in outputs and hasattr(batch['bill'], 'y'):
            success_logits = outputs['success_logit']
            success_targets = torch.nan_to_num(batch['bill'].y[batch['bill'].node_id], nan=0.0).float()
            success_loss = F.binary_cross_entropy_with_logits(success_logits, success_targets)
        loss_dict['success_loss'] = success_loss

        influence_reg = torch.tensor(0.0, device=self.device)
        if 'influence_dict' in outputs:
            for nt, inf in outputs['influence_dict'].items():
                influence_reg += outputs['influence_dict'][nt].mean()
        loss_dict['influence_reg'] = influence_reg

        total_loss = (topic_loss + link_loss + success_loss + influence_reg)
        loss_dict['total_loss'] = total_loss

        return loss_dict

class StableLossComputer:
    def __init__(self, device=device, temp=0.1, neg_sampling_ratio=1):
        self.device = device
        self.temp = temp
        self.neg_sampling_ratio = neg_sampling_ratio
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = 1e-8

    def compute_topic_loss(self, logits, labels):
        logits = torch.clamp(logits, -10, 10)

        num_classes = logits.size(-1)
        smooth_labels = F.one_hot(labels, num_classes).float()
        smooth_labels = smooth_labels * 0.9 + 0.1 / num_classes

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

        return safe_tensor_operation(lambda: loss, default_value=0.0).clamp(0, 1000)

    def compute_link_loss(self, embeddings, edge_index_dict):
        total_loss = 0.0
        num_edges = 0

        for (s_t, _, d_t), ei in edge_index_dict.items():
            if ei.numel() == 0:
                continue
            s, d = ei
            if s.numel() == 0 or d.numel() == 0:
                continue
            z_s = embeddings[s_t][s]
            z_d = embeddings[d_t][d]
            if torch.isnan(z_s).any() or torch.isinf(z_s).any():
                continue
            if torch.isnan(z_d).any() or torch.isinf(z_d).any():
                continue
            z_s = F.normalize(torch.clamp(z_s, -10, 10), dim=-1, eps=1e-8)
            z_d = F.normalize(torch.clamp(z_d, -10, 10), dim=-1, eps=1e-8)

            pos_scores = torch.clamp((z_s * z_d).sum(-1) / self.temp, -10, 10)
            pos_loss = self.bce(pos_scores, torch.ones_like(pos_scores))


            if self.neg_sampling_ratio > 0:
                n_neg = min(s.size(0) * self.neg_sampling_ratio, embeddings[d_t].size(0))
                neg_indices = torch.randperm(embeddings[d_t].size(0), device=self.device)[:n_neg]

                z_s_neg = z_s[:n_neg] if n_neg <= s.size(0) else z_s.repeat(n_neg // s.size(0) + 1, 1)[:n_neg]
                z_d_neg = F.normalize(torch.clamp(embeddings[d_t][neg_indices], -10, 10), dim=-1, eps=1e-8)

                neg_scores = torch.clamp((z_s_neg * z_d_neg).sum(-1) / self.temp, -10, 10)
                neg_loss = self.bce(neg_scores, torch.zeros_like(neg_scores))

                if not torch.isnan(neg_loss) and not torch.isinf(neg_loss):
                    total_loss += pos_loss + neg_loss
                else:
                    total_loss += pos_loss
            else:
                total_loss += pos_loss

            num_edges += 1

        return torch.clamp(total_loss / max(num_edges, 1), 0, 1000)
def _adj(data, et):
    row, col = data.edge_index_dict[et]
    ns, nd = data[et[0]].num_nodes, data[et[2]].num_nodes
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.ones_like(row, dtype=torch.float32), (ns, nd)
    ).coalesce()

def build_masks_weights(d, acts, A_by, paths, device=device):
    dev = torch.device('cpu')
    rows = {nt: d[nt].node_id.cpu() for nt in acts}
    cols = d['bill_version'].node_id.cpu()
    out = {}
    col_map = {int(g): i for i, g in enumerate(cols.tolist())}
    for nt in acts:
        n_ids = rows[nt]
        row_map = {int(g): i for i, g in enumerate(n_ids.tolist())}
        n_r = n_ids.size(0)
        n_c = cols.size(0)
        m = torch.zeros(n_r, n_c, dtype=torch.bool, device=dev)
        w = torch.zeros(n_r, n_c, dtype=torch.float32, device=dev)

        for path in paths[nt]:
            S = A_by[path[0]]
            for et in path[1:]:
                S = (S @ A_by[et]).coalesce()

            r, c = S.indices()
            keep = torch.isin(r, rows[nt]) & torch.isin(c, cols)
            if not keep.any():
                continue
            r_kept = r[keep].tolist()
            c_kept = c[keep].tolist()
            r_sub = [row_map[int(g)] for g in r_kept]
            c_sub = [col_map[int(g)] for g in c_kept]
            m[r_sub, c_sub] = True
            if any(e in path[0][1] or e in path[1][1] for e in {'donated_to','wrote','lobbied','member_of'}):
                ei = A_by[path[0]].indices()
                ea = A_by[path[0]].values()
                Wsp = torch.sparse_coo_tensor(ei, ea, A_by[path[0]].shape, device=dev).coalesce()
                for et in path[1:]:
                    Wsp = (Wsp @ A_by[et]).coalesce()

                wr, wc = Wsp.indices()
                keep_w = torch.isin(wr, rows[nt]) & torch.isin(wc, cols)
                if keep_w.any():
                    wr_sub = [row_map[int(g)] for g in wr[keep_w].tolist()]
                    wc_sub = [col_map[int(g)] for g in wc[keep_w].tolist()]
                    w[wr_sub, wc_sub] += Wsp.values()[keep_w]

        out[nt] = (m.to(device), w.to(device))
    return out

class FinalOutputSaver:
    def __init__(self, checkpoint_path, model, data, loader, A_by, ALLOWED, device = "mps", out_dir = "outputs"):
        self.device = torch.device(device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval().to(self.device)
        self.model = model

        h_dim = hidden_dim
        k_topics = model.topic_onehot.size(1)

        self.emb_sum = {
            nt: torch.zeros(data[nt].num_nodes, h_dim, dtype=torch.float32)
            for nt in data.node_types
        }
        self.emb_cnt = {
            nt: torch.zeros(data[nt].num_nodes, dtype=torch.long)
            for nt in data.node_types
        }

        n_bill = data["bill"].num_nodes
        self.bill_logit_sum = torch.zeros(n_bill, k_topics)
        self.bill_logit_cnt = torch.zeros(n_bill, dtype=torch.long)
        self.success_sum = torch.zeros(n_bill)
        self.success_cnt = torch.zeros(n_bill, dtype=torch.long)

        self.actor_align_sum = {
            nt: torch.zeros(data[nt].num_nodes, k_topics)
            for nt in model.actor_types
        }
        self.actor_align_cnt = {
            nt: torch.zeros(data[nt].num_nodes, dtype=torch.long)
            for nt in model.actor_types
        }
        self.actor_infl_sum = {
            nt: torch.zeros(data[nt].num_nodes)
            for nt in model.actor_types
        }
        self.actor_infl_cnt = {
            nt: torch.zeros(data[nt].num_nodes, dtype=torch.long)
            for nt in model.actor_types
        }

        self.loader = loader
        self.A_by = A_by
        self.ALLOWED = ALLOWED

    @torch.no_grad()
    def run(self):
        for batch in tqdm(self.loader, desc="Saving final outputs"):
            batch = batch.to(self.device, non_blocking=True)
            mw_dict = build_masks_weights(
                batch, self.model.actor_types, self.A_by, self.ALLOWED, device=self.device
            )
            out = self.model(batch, mw_dict)

            for nt, z_nt in out["node_embeddings"].items():
                nids = batch[nt].node_id.cpu()
                z_cpu = z_nt.cpu()
                self.emb_sum[nt].index_add_(0, nids, z_cpu)
                self.emb_cnt[nt].index_add_(0, nids, torch.ones_like(nids))

            bill_ids = batch["bill"].node_id.cpu()
            self.bill_logit_sum.index_add_(0, bill_ids, out["bill_logits"].cpu())
            self.bill_logit_cnt.index_add_(0, bill_ids, torch.ones_like(bill_ids))
            self.success_sum.index_add_(0, bill_ids, out["success_logit"].cpu())
            self.success_cnt.index_add_(0, bill_ids, torch.ones_like(bill_ids))

            for nt in self.model.actor_types:
                if nt not in out["actor_topic_dict"]:
                    continue
                nids = batch[nt].node_id.cpu()
                self.actor_align_sum[nt].index_add_(0, nids, out["actor_topic_dict"][nt].cpu())
                self.actor_align_cnt[nt].index_add_(0, nids, torch.ones_like(nids))
                self.actor_infl_sum[nt].index_add_(0, nids, out["influence_dict"][nt].cpu())
                self.actor_infl_cnt[nt].index_add_(0, nids, torch.ones_like(nids))

        emb_final = {
            nt: self.emb_sum[nt] / self.emb_cnt[nt].clamp_min(1).unsqueeze(-1)
            for nt in self.emb_sum
        }
        bill_logits_final = self.bill_logit_sum / self.bill_logit_cnt.clamp_min(1).unsqueeze(-1)
        success_final = self.success_sum / self.success_cnt.clamp_min(1)

        actor_align_final = {
            nt: self.actor_align_sum[nt] / self.actor_align_cnt[nt].clamp_min(1).unsqueeze(-1)
            for nt in self.actor_align_sum
        }
        actor_infl_final = {
            nt: self.actor_infl_sum[nt] / self.actor_infl_cnt[nt].clamp_min(1)
            for nt in self.actor_infl_sum
        }

        torch.save(emb_final, self.out_dir / "node_embeddings.pt")
        torch.save(
            {
                "bill_logits": bill_logits_final,
                "success_logit": success_final,
                "actor_align": actor_align_final,
                "actor_influence": actor_infl_final,
            },
            self.out_dir / "predictions.pt",
        )

def main():
    data = load_and_preprocess_data()
    logger = setup_logging()

    with open('bill_labels_updated.json', 'r') as f:
        topic_cluster_labels_dict = json.load(f)

    num_topics = len(list(set([v for v in topic_cluster_labels_dict.values()]))) + 1
    cluster_bill = torch.full((data['bill'].num_nodes,), num_topics, dtype=torch.long)

    key1 = data['bill'].n_id.tolist()
    key2 = data['bill'].node_id.tolist()
    key = {k1: k2 for k1, k2 in zip(key1, key2)}
    for bill_nid, lab in topic_cluster_labels_dict.items():
        if bill_nid in key:
            cluster_bill[key[bill_nid]] = lab

    topic_onehot_bill = F.one_hot(cluster_bill.clamp(max=num_topics-1), num_classes=num_topics).float()
    cluster_bill, topic_onehot_bill = cluster_bill.to(device), topic_onehot_bill.to(device)
    A_by = {et: _adj(data, et).coalesce() for et in data.edge_types}

    ALLOWED = {
        "donor": [
            ( ('donor','donated_to','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            (
            ('donor', 'donated_to','legislator_term'),
            ('legislator_term','wrote','bill_version') ),
        ],
        "lobby_firm": [
            ( ('lobby_firm','lobbied','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('lobby_firm','lobbied','committee'),
            ('committee','rev_member_of','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('lobby_firm','lobbied','committee'),
            ('committee','rev_member_of','legislator_term'),
            ('legislator_term','wrote','bill_version') ),
            ( ('lobby_firm','lobbied','legislator_term'),
            ('legislator_term', 'wrote','bill_version') )
        ],
        "committee": [
            ( ('committee','rev_member_of','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('committee','rev_member_of','legislator_term'),
            ('legislator_term','wrote','bill_version') )
        ],
        "legislator": [
            ( ('legislator', 'samePerson', 'legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('legislator', 'samePerson', 'legislator_term'),
            ('legislator_term','wrote','bill_version') )
        ]
    }
    in_dims = {nt: data[nt].x.size(1) for nt in data.node_types if hasattr(data[nt], 'x') and data[nt].x is not None}

    metadata = data.metadata()
    relation_weight_sharing = DEFAULT_RELATION_WEIGHT_SHARING

    model = LegislativeGraphModel(
        in_dims,
        cluster_bill,
        topic_onehot_bill,
        hidden_dim,
        dropout_p,
        metadata,
        relation_weight_sharing=relation_weight_sharing,
        device=device,
    ).to(device)

    torch.mps.empty_cache()
    gc.collect()

    for nt in data.node_types:
        if hasattr(data[nt], 'n_id'):
            delattr(data[nt], 'n_id')

    loader = HGTLoader(
        data,
        num_samples={
            'legislator_term': [240] * 2,
            'legislator': [240] * 2,
            'committee': [212] * 2,
            'lobby_firm': [212] * 2,
            'donor': [212] * 2,
            'bill_version': [9600] * 2,
            'bill': [4800] * 2,
        },
        batch_size=248,
        shuffle=True,
        input_nodes='legislator_term'
    )

    saver = FinalOutputSaver(
        checkpoint_path='GNN/checkpoint_epoch_3.pt',
        model=model,
        data=data,
        loader=loader,
        A_by=A_by,
        ALLOWED=ALLOWED,
        out_dir='dashboard'
    )
    saver.run()

if __name__ == "__main__":
    main()