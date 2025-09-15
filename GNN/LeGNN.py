import datetime, gc, json, torch, logging, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv
from contextlib import contextmanager
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
    # Detect whether mapping is edge_type -> group or group -> [edge_types]
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

def safe_scatter_mean(src, index, dim_size):
    ones = torch.ones(src.size(0), dtype=src.dtype, device=src.device)
    sum_ = scatter_add(src, index, dim=0, dim_size=dim_size)
    count = scatter_add(ones, index, dim=0, dim_size=dim_size).clamp(min=1)
    return sum_ / count.unsqueeze(-1)
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

def build_valid_destination_sets(data):
    """Collect the set of valid destination node ids for every edge type."""

    valid_sets = {}
    for edge_type in data.edge_types:
        edge_index = data.edge_index_dict[edge_type]
        dst_nodes = edge_index[1]
        if isinstance(dst_nodes, torch.Tensor):
            dst_nodes = dst_nodes.detach()
            if dst_nodes.device.type != 'cpu':
                dst_nodes = dst_nodes.to('cpu')
            dst_nodes = dst_nodes.to(dtype=torch.long)
        else:
            dst_nodes = torch.as_tensor(dst_nodes, dtype=torch.long)

        valid_sets[tuple(edge_type)] = torch.unique(dst_nodes)

    return valid_sets

def compute_relation_neg_sampling_ratios(data, *, min_ratio=1, max_ratio=5):
    """Estimate negative sampling ratios per edge type based on sparsity."""

    avg_degree_by_edge = {}
    degree_values = []

    for edge_type in data.edge_types:
        edge_index = data.edge_index_dict[edge_type]
        edge_key = tuple(edge_type)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            avg_degree_by_edge[edge_key] = 0.0
            continue

        src_nodes = edge_index[0].detach()
        if src_nodes.device.type != 'cpu':
            src_nodes = src_nodes.to('cpu')
        src_nodes = src_nodes.to(dtype=torch.long)

        unique_sources = torch.unique(src_nodes).numel()
        unique_sources = max(int(unique_sources), 1)
        avg_degree = float(num_edges) / unique_sources

        avg_degree_by_edge[edge_key] = avg_degree
        degree_values.append(avg_degree)

    global_avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0.0
    default_ratio = max(
        min_ratio,
        min(max_ratio, int(math.ceil(global_avg_degree)) if global_avg_degree > 0 else min_ratio),
    )

    ratios = {}
    for edge_key, avg_degree in avg_degree_by_edge.items():
        if avg_degree <= 0.0:
            ratios[edge_key] = 0
            continue

        ratio = global_avg_degree / avg_degree if global_avg_degree > 0 else 1.0
        ratio = max(min_ratio, min(max_ratio, int(math.ceil(ratio))))
        ratios[edge_key] = ratio

    ratios['default'] = default_ratio
    return ratios

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
    """Compute controversy scores for bill versions based on vote signals.

    Parameters
    ----------
    data:
        Heterogeneous graph data containing a ``('legislator_term', 'voted_on',
        'bill_version')`` edge type with a vote signal stored in
        ``edge_attr``.
    session_attr:
        Optional attribute (name or tensor) that encodes the legislative
        session or year for each bill version. When provided, the function
        will aggregate controversy statistics per session in addition to the
        per-bill values.
    total_possible_attr:
        Attribute (name, tensor, or mapping) describing the total possible
        votes for each bill version (for example, committee size or chamber
        membership). When omitted, the function falls back to the observed
        vote counts (yes + no + abstain).
    """

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
class TGATRelativeTimeEncoder(nn.Module):
    def __init__(self, hidden_dim, in_channels=5, num_frequencies=16):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_frequencies = max(1, min(num_frequencies, hidden_dim // 2))

        self.frequencies = nn.Parameter(torch.randn(in_channels, self.num_frequencies))
        self.phase = nn.Parameter(torch.zeros(self.num_frequencies))
        self.proj = nn.Sequential(
            nn.Linear(self.num_frequencies * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_features):
        if time_features.dim() == 1:
            time_features = time_features.unsqueeze(-1)

        features = torch.nan_to_num(time_features, nan=0.0, posinf=0.0, neginf=0.0)
        if features.size(-1) != self.in_channels:
            pad = self.in_channels - features.size(-1)
            if pad > 0:
                padding = torch.zeros(*features.shape[:-1], pad, device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :self.in_channels]

        max_abs = features.abs().amax(dim=0, keepdim=True).clamp(min=1.0)
        scaled = features / max_abs

        angles = scaled @ self.frequencies + self.phase
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        encoding = torch.cat([sin_part, cos_part], dim=-1)
        return self.proj(encoding)


class LegislativeTemporalEncoder(nn.Module):
    def __init__(self, d=hidden_dim, in_channels=5, num_frequencies=16):
        super().__init__()
        self.hidden_dim = d
        self.time_encoder = TGATRelativeTimeEncoder(d, in_channels=in_channels, num_frequencies=num_frequencies)

        def _process_block():
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )

        self.process_blocks = nn.ModuleDict({
            'default': _process_block(),
            'vote': _process_block(),
            'donation': _process_block(),
            'lobbying': _process_block(),
            'hierarchy': _process_block(),
        })

    def forward(self, time_features, process_type):
        encoded = self.time_encoder(time_features)
        block = self.process_blocks.get(process_type, self.process_blocks['default'])
        return block(encoded)

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
        dk = d // h
        self.Q = nn.Linear(d, d, bias=False)
        self.K = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        self.dk = dk

    def forward(self, a_z, bv_z, mask, weight=None):
        if torch.isnan(a_z).any() or torch.isinf(a_z).any():
            a_z = torch.nan_to_num(a_z, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(bv_z).any() or torch.isinf(bv_z).any():
            bv_z = torch.nan_to_num(bv_z, nan=0.0, posinf=1.0, neginf=-1.0)

        q = self.Q(a_z).view(-1, self.h, self.dk).transpose(0, 1)
        k = self.K(bv_z).view(-1, self.h, self.dk).transpose(0, 1)
        v = self.V(bv_z).view(-1, self.h, self.dk).transpose(0, 1)

        att = (q @ k.transpose(-1, -2)) / np.sqrt(self.dk)
        att = att.transpose(0, 1)

        if mask.numel() == 0 or (~mask).all():
            context = torch.zeros_like(v.transpose(0, 1))
            topic_align = context.mean(0)
            influence = torch.zeros(mask.size(0), device=mask.device)
            return topic_align, influence

        att = att.masked_fill(~mask.unsqueeze(1), -1e4)

        if weight is not None:
            weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=-1.0)
            att = att + weight.unsqueeze(1)

        att = torch.clamp(att, -10, 10)
        att_max = att.max(dim=-1, keepdim=True)[0]
        att_stable = att - att_max
        p = F.softmax(att_stable, dim=-1)
        if torch.isnan(p).any():
            p = torch.nan_to_num(p, nan=1.0/p.size(-1))
            p = p / p.sum(dim=-1, keepdim=True)

        context = torch.matmul(p.transpose(0, 1), v)
        topic_align = context.mean(0)
        influence = p.mean(1).sum(-1)
        influence = torch.nan_to_num(influence, nan=0.0, posinf=1.0, neginf=-1.0)
        return topic_align, influence

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

        donation_edges = [
            ('donor', 'donated_to', 'legislator_term'),
            ('legislator_term', 'rev_donated_to', 'donor'),
        ]
        lobbying_edges = [
            ('lobby_firm', 'lobbied', 'legislator_term'),
            ('lobby_firm', 'lobbied', 'committee'),
            ('committee', 'rev_lobbied', 'lobby_firm'),
            ('legislator_term', 'rev_lobbied', 'lobby_firm'),
        ]
        vote_edges = [
            ('legislator_term', 'voted_on', 'bill_version'),
            ('bill_version', 'rev_voted_on', 'legislator_term'),
        ]
        hierarchy_edges = [
            ('bill_version', 'is_version', 'bill'),
            ('bill', 'rev_is_version', 'bill_version'),
        ]

        process_map = {}
        for group, label in [
            (donation_edges, 'donation'),
            (lobbying_edges, 'lobbying'),
            (vote_edges, 'vote'),
            (hierarchy_edges, 'hierarchy'),
        ]:
            for et in group:
                process_map[tuple(et)] = label
                process_map[et[1]] = label

        self.process_map = process_map

        self.convs = nn.ModuleDict()
        for group_name, relations in self.relation_groups.items():
            metadata_subset = (self.metadata[0], relations)
            self.convs[group_name] = HGTConv(hidden_dim, hidden_dim, metadata_subset, heads=self.heads)

        self.temporal_encoder = LegislativeTemporalEncoder(hidden_dim)
        self.vote_conv = PolarityAwareConv(hidden_dim, 385, dropout)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        edge_ts_dict=None,
        node_ts_dict=None,
        edge_delta_dict=None,
    ):
        out = {nt: torch.zeros_like(x, device=x.device) for nt, x in x_dict.items()}

        edge_ts_dict = edge_ts_dict or {}
        node_ts_dict = node_ts_dict or {}
        edge_delta_dict = edge_delta_dict or {}

        node_time_tensors = {}
        for nt, ts in node_ts_dict.items():
            if ts is None:
                continue
            tensor = sanitize(ts)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            node_time_tensors[nt] = tensor.view(-1)

        edge_delta_tensors = {}
        for et, delta in edge_delta_dict.items():
            if delta is None:
                continue
            tensor = sanitize(delta)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            edge_delta_tensors[_normalize_edge_type(et)] = tensor.view(-1)

        temporal_messages = {}

        for et, timestamps in edge_ts_dict.items():
            if timestamps is None:
                continue

            norm_et = _normalize_edge_type(et)
            if norm_et not in edge_index_dict:
                continue

            edge_index = edge_index_dict[norm_et]
            if edge_index.numel() == 0:
                continue

            src_type, _, dst_type = norm_et
            if dst_type not in x_dict:
                continue

            num_edges = edge_index.size(1)

            ts_tensor = sanitize(timestamps).view(-1)
            if ts_tensor.numel() == 0:
                continue

            ts_tensor = ts_tensor.to(x_dict[dst_type].device)
            if ts_tensor.numel() != num_edges:
                min_len = min(ts_tensor.numel(), num_edges)
                ts_tensor = ts_tensor[:min_len]
                if min_len < num_edges:
                    pad_value = ts_tensor.mean() if min_len > 0 else torch.tensor(0.0, device=ts_tensor.device)
                    pad_scalar = float(pad_value.item()) if isinstance(pad_value, torch.Tensor) else float(pad_value)
                    ts_tensor = F.pad(ts_tensor, (0, num_edges - min_len), value=pad_scalar)

            src_times = node_time_tensors.get(src_type)
            dst_times = node_time_tensors.get(dst_type)

            if src_times is not None:
                src_times = src_times.to(ts_tensor.device)
            if dst_times is not None:
                dst_times = dst_times.to(ts_tensor.device)

            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            base_time = torch.nan_to_num(ts_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            src_delta = torch.zeros_like(base_time)
            if src_times is not None and src_times.numel() > 0:
                valid_src = src_idx < src_times.size(0)
                if valid_src.any():
                    src_delta[valid_src] = base_time[valid_src] - src_times[src_idx[valid_src]]
            else:
                valid_src = torch.zeros_like(base_time, dtype=torch.bool)

            dst_delta = torch.zeros_like(base_time)
            if dst_times is not None and dst_times.numel() > 0:
                valid_dst = dst_idx < dst_times.size(0)
                if valid_dst.any():
                    dst_delta[valid_dst] = dst_times[dst_idx[valid_dst]] - base_time[valid_dst]
            else:
                valid_dst = torch.zeros_like(base_time, dtype=torch.bool)

            src_dst_delta = torch.zeros_like(base_time)
            if src_times is not None and dst_times is not None and src_times.numel() > 0 and dst_times.numel() > 0:
                valid_pair = valid_src & valid_dst
                if valid_pair.any():
                    src_dst_delta[valid_pair] = dst_times[dst_idx[valid_pair]] - src_times[src_idx[valid_pair]]

            delta_feature = torch.zeros_like(base_time)
            delta_tensor = edge_delta_tensors.get(norm_et)
            if delta_tensor is None:
                delta_tensor = edge_delta_tensors.get(_et_to_key(norm_et))
            if delta_tensor is not None:
                delta_tensor = delta_tensor.to(base_time.device)
                if delta_tensor.numel() != num_edges:
                    min_len = min(delta_tensor.numel(), num_edges)
                    delta_tensor = delta_tensor[:min_len]
                    if min_len < num_edges:
                        pad_value = delta_tensor.mean() if min_len > 0 else torch.tensor(0.0, device=delta_tensor.device)
                        pad_scalar = float(pad_value.item()) if isinstance(pad_value, torch.Tensor) else float(pad_value)
                        delta_tensor = F.pad(delta_tensor, (0, num_edges - min_len), value=pad_scalar)
                delta_feature = torch.nan_to_num(delta_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            time_features = torch.stack([
                base_time,
                src_delta,
                dst_delta,
                src_dst_delta,
                delta_feature,
            ], dim=-1)

            process = self.process_map.get(norm_et)
            if process is None:
                process = self.process_map.get(norm_et[1], 'vote')

            encoded = self.temporal_encoder(time_features, process)
            encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)

            aggregated = scatter_mean(encoded, dst_idx, dim=0, dim_size=x_dict[dst_type].size(0))
            temporal_messages[norm_et] = torch.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)

        remaining_temporal = dict(temporal_messages)

        for group_name, conv in self.convs.items():
            relations = self.relation_groups[group_name]
            group_edge_index = {}

            for et in relations:
                if et not in edge_index_dict:
                    continue
                edge_index = edge_index_dict[et]
                if edge_index.numel() == 0:
                    continue
                group_edge_index[et] = edge_index

            if not group_edge_index:
                continue

            conv_out = conv(x_dict, group_edge_index)
            for nt, value in conv_out.items():
                cleaned = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                if nt in out:
                    out[nt] += cleaned
                else:
                    out[nt] = cleaned

            for et in relations:
                addition = remaining_temporal.pop(et, None)
                if addition is None:
                    continue
                dst_type = et[2]
                if dst_type in out:
                    out[dst_type] += addition
                else:
                    out[dst_type] = addition

        for et, addition in remaining_temporal.items():
            dst_type = et[2]
            if dst_type in out:
                out[dst_type] += addition
            else:
                out[dst_type] = addition

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
    def __init__(
        self,
        in_dims,
        cluster_id,
        topic_onehot,
        hidden_dim,
        dropout,
        metadata,
        relation_weight_sharing=None,
        device=device,
        neg_sampling_ratio=1,
        valid_destination_sets=None,
    ):
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

        self.bill_alpha = nn.Parameter(torch.tensor(0.5))
        self.leg_alpha  = nn.Parameter(torch.tensor(0.5))

        self.topic_head = BillTopicHead(hidden_dim, topic_onehot.size(1))
        self.success = SuccessHead(hidden_dim)

        self.actor_types = ['legislator', 'committee', 'donor', 'lobby_firm']
        self.actor_head = nn.ModuleDict({nt: ActorHead(hidden_dim) for nt in self.actor_types})

        k_topics = topic_onehot.size(1)
        dk = hidden_dim // self.actor_head['legislator'].h
        self.actor_topic_proj = nn.ModuleDict({
            nt: nn.Linear(dk, k_topics, bias=False) for nt in self.actor_types
        })

        self.register_buffer('cluster_id',  cluster_id)
        self.register_buffer('topic_onehot', topic_onehot)
        self.loss_computer = StableLossComputer(
            device=device,
            neg_sampling_ratio=neg_sampling_ratio,
            valid_destination_sets=valid_destination_sets,
        )

    def _aggregate_hierarchy(self, z, b):
        if ('bill_version', 'is_version', 'bill') in b.edge_index_dict:
            s, d = b.edge_index_dict[('bill_version', 'is_version', 'bill')]
            if s.numel() > 0 and d.numel() > 0:
                bill_agg = safe_scatter_mean(z['bill_version'][s], d, dim_size=z['bill'].size(0))
                if torch.isnan(bill_agg).any():
                    bill_agg = torch.zeros_like(bill_agg)
                z['bill'] = self.bill_alpha * z['bill'] + (1 - self.bill_alpha) * bill_agg

        if ('legislator', 'samePerson', 'legislator_term') in b.edge_index_dict:
            d2, s2 = b.edge_index_dict[('legislator', 'samePerson', 'legislator_term')]
            if s2.numel() > 0 and d2.numel() > 0:
                leg_agg = safe_scatter_mean(z['legislator_term'][s2], d2, dim_size=z['legislator'].size(0))
                if torch.isnan(leg_agg).any():
                    leg_agg = torch.zeros_like(leg_agg)
                z['legislator'] = self.leg_alpha * z['legislator'] + (1 - self.leg_alpha) * leg_agg

        return z

    def encoder(self, batch):
        x_dict = {
            nt: self.feature_proj[nt](batch[nt].x)
            for nt in self.feature_proj if hasattr(batch[nt], 'x')
        }

        edge_ts_dict = {}
        edge_delta_dict = {}
        for et in batch.edge_types:
            store = batch[et]
            if hasattr(store, 'timestamp') and getattr(store, 'timestamp') is not None:
                edge_ts_dict[et] = store.timestamp

            for attr_name in ('time_diff', 'time_diffs', 'time_delta', 'time_deltas', 'delta_time'):
                if hasattr(store, attr_name):
                    value = getattr(store, attr_name)
                    if value is not None:
                        edge_delta_dict[et] = value
                        break

        node_ts_dict = {}
        if hasattr(batch, 'node_types'):
            node_iter = batch.node_types
        else:
            node_iter = self.node_types

        for nt in node_iter:
            if not hasattr(batch, nt):
                continue
            store = batch[nt]
            if hasattr(store, 'timestamp') and getattr(store, 'timestamp') is not None:
                node_ts_dict[nt] = store.timestamp

        for encoder in self.encoders:
            x_dict = encoder(
                x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                edge_ts_dict,
                node_ts_dict,
                edge_delta_dict,
            )

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
        link_loss = self.loss_computer.compute_link_loss(embeddings, batch)
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
    def __init__(self, device=device, temp=0.1, neg_sampling_ratio=1, valid_destination_sets=None):
        self.device = device
        self.temp = temp
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = 1e-8
        self._set_neg_sampling_ratio(neg_sampling_ratio)
        self.valid_destination_sets = self._prepare_valid_destination_sets(valid_destination_sets)

    def _set_neg_sampling_ratio(self, ratio):
        self.default_neg_sampling_ratio = 1
        self.neg_sampling_ratio_map = {}

        if isinstance(ratio, Mapping):
            default_value = None
            for key, value in ratio.items():
                if isinstance(key, str) and key.lower() == 'default':
                    default_value = value
                    continue
                try:
                    edge_key = _normalize_edge_type(key)
                except (TypeError, ValueError):
                    continue
                try:
                    ratio_value = int(value)
                except (TypeError, ValueError):
                    continue
                self.neg_sampling_ratio_map[edge_key] = max(0, ratio_value)

            if default_value is not None:
                try:
                    self.default_neg_sampling_ratio = max(0, int(default_value))
                except (TypeError, ValueError):
                    self.default_neg_sampling_ratio = 0
        else:
            try:
                self.default_neg_sampling_ratio = max(0, int(ratio))
            except (TypeError, ValueError):
                self.default_neg_sampling_ratio = 0

    def _prepare_valid_destination_sets(self, mapping):
        if not mapping:
            return {}

        prepared = {}
        for key, value in mapping.items():
            try:
                edge_key = _normalize_edge_type(key)
            except (TypeError, ValueError):
                continue

            if value is None:
                continue

            tensor = torch.as_tensor(value, dtype=torch.long)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            tensor = torch.unique(tensor)
            if tensor.device.type != 'cpu':
                tensor = tensor.to('cpu')

            prepared[edge_key] = tensor

        return prepared

    def _get_neg_sampling_ratio(self, edge_type):
        edge_key = tuple(edge_type)
        return self.neg_sampling_ratio_map.get(edge_key, self.default_neg_sampling_ratio)

    def _candidate_negative_indices(self, edge_type, embeddings, batch, dst_type):
        num_nodes = embeddings[dst_type].size(0)
        if num_nodes == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        candidates = torch.arange(num_nodes, device=self.device, dtype=torch.long)

        if not self.valid_destination_sets:
            return candidates

        valid_global = self.valid_destination_sets.get(edge_type)
        if valid_global is None:
            return candidates
        if valid_global.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        if hasattr(batch, 'node_types') and dst_type not in batch.node_types:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        node_store = batch[dst_type]
        node_ids = getattr(node_store, 'node_id', None)
        if node_ids is None:
            return candidates

        node_ids_cpu = node_ids.detach()
        if node_ids_cpu.device.type != 'cpu':
            node_ids_cpu = node_ids_cpu.to('cpu')
        node_ids_cpu = node_ids_cpu.to(dtype=torch.long)

        mask_cpu = torch.isin(node_ids_cpu, valid_global)
        if not mask_cpu.any():
            return torch.zeros(0, dtype=torch.long, device=self.device)

        return candidates[mask_cpu.to(device=candidates.device)]

    def compute_topic_loss(self, logits, labels):
        logits = torch.clamp(logits, -10, 10)

        num_classes = logits.size(-1)
        smooth_labels = F.one_hot(labels, num_classes).float()
        smooth_labels = smooth_labels * 0.9 + 0.1 / num_classes

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

        return safe_tensor_operation(lambda: loss, default_value=0.0).clamp(0, 1000)

    def compute_link_loss(self, embeddings, batch):
        total_loss = 0.0
        num_edges = 0

        for edge_type, edge_index in batch.edge_index_dict.items():
            if edge_index.numel() == 0:
                continue

            s, d = edge_index
            if s.numel() == 0 or d.numel() == 0:
                continue

            s_t, _, d_t = edge_type
            if s_t not in embeddings or d_t not in embeddings:
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
            relation_loss = pos_loss

            neg_ratio = self._get_neg_sampling_ratio(edge_type)
            if neg_ratio > 0:
                candidates = self._candidate_negative_indices(edge_type, embeddings, batch, d_t)
                if candidates.numel() > 0:
                    max_candidates = int(candidates.numel())
                    n_neg = min(s.size(0) * neg_ratio, max_candidates)
                    if n_neg > 0:
                        perm = torch.randperm(max_candidates, device=self.device)[:n_neg]
                        neg_indices = candidates[perm]

                        if z_s.size(0) >= n_neg:
                            z_s_neg = z_s[:n_neg]
                        else:
                            repeat = math.ceil(n_neg / z_s.size(0))
                            z_s_neg = z_s.repeat(repeat, 1)[:n_neg]

                        z_d_neg = F.normalize(
                            torch.clamp(embeddings[d_t][neg_indices], -10, 10),
                            dim=-1,
                            eps=1e-8,
                        )

                        neg_scores = torch.clamp((z_s_neg * z_d_neg).sum(-1) / self.temp, -10, 10)
                        neg_loss = self.bce(neg_scores, torch.zeros_like(neg_scores))

                        if not torch.isnan(neg_loss) and not torch.isinf(neg_loss):
                            relation_loss = relation_loss + neg_loss

            total_loss += relation_loss
            num_edges += 1

        try:
            return torch.clamp(total_loss / max(num_edges, 1), 0, 1000)
        except Exception:
            return torch.tensor(0.0, device=self.device)

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

class TrainingManager:
    def __init__(self, model, optimizer, logger, A_by, ALLOWED):
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.loss_computer = StableLossComputer()
        self.A_by = A_by
        self.ALLOWED = ALLOWED

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.epoch_metrics = []

    def train_epoch(self, loader, epoch):
        self.model.train()
        epoch_losses = []

        with error_context(f"Training epoch {epoch}", self.logger):
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
                try:
                    loss = self.train_step(batch)
                    print(loss)
                    epoch_losses.append(loss)
                    if batch_idx % 10 == 0:
                        torch.mps.empty_cache()
                        gc.collect()

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        self.train_losses.append(avg_loss)

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.patience_counter = 0
            self.save_checkpoint(epoch, avg_loss)
        else:
            self.patience_counter += 1

        return avg_loss

    def train_step(self, batch):
        batch = batch.to(device)
        self.optimizer.zero_grad()
        mw_dict = build_masks_weights(batch, ['legislator','committee','donor','lobby_firm'], self.A_by, self.ALLOWED)

        outputs = self.model(batch, mw_dict)
        loss_dict = self.model.compute_loss(outputs, batch)
        total_loss = loss_dict['total_loss']
        print(loss_dict)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"Invalid loss value: {total_loss}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return total_loss.item()

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
        self.logger.info(f"Saved checkpoint at epoch {epoch} with loss {loss:.4f}")


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
    valid_destination_sets = build_valid_destination_sets(data)
    neg_sampling_ratios = compute_relation_neg_sampling_ratios(data)

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
        neg_sampling_ratio=neg_sampling_ratios,
        valid_destination_sets=valid_destination_sets,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, eps=1e-8)

    torch.mps.empty_cache()
    gc.collect()

    for nt in data.node_types:
        if hasattr(data[nt], 'n_id'):
            delattr(data[nt], 'n_id')

    loader = HGTLoader(
        data,
        num_samples={
            'legislator_term': [212] * 3,
            'legislator': [240] * 3,
            'committee': [212] * 3,
            'lobby_firm': [212] * 3,
            'donor': [212] * 3,
            'bill_version': [9600] * 3,
            'bill': [4800] * 3,
        },
        batch_size=248,
        shuffle=True,
        input_nodes='legislator_term'
    )

    epochs = 50
    patience = 5
    # model.load_state_dict(torch.load("best_model5.pt"), strict=False)
    # optimizer.load_state_dict(torch.load("best_opt5.pt"))
    trainer = TrainingManager(optimizer=optimizer, logger=logger, model=model, A_by=A_by, ALLOWED=ALLOWED)

    for epoch in tqdm(range(epochs), position=0, total=epochs):
        avg_loss = trainer.train_epoch(loader, epoch)
        logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        if trainer.patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    logger.info("Training completed")
if __name__ == "__main__":
    main()