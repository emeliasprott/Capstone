import datetime, gc, json, torch, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import SAGEConv
from contextlib import contextmanager
from GNN.time_utils import convert_to_utc_seconds as _convert_to_utc_seconds_list

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

hidden_dim = 192
n_layers = 3
dropout_p = 0.10
device = 'mps'
inf_reg_weight = 0.1
ent_reg_weight = 0.1

def _init_linear(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=0.01)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def sanitize(t, clamp=1e4):
    t = t.float() if t.dtype == torch.float64 else t
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
    return t.clamp_(-clamp, clamp)

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
def clean_features(data):
    for nt in data.node_types:
        x = data[nt].x
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True).clamp(min=1e-5)
        x = ((x - mean) / std).clamp(-10, 10)
        data[nt].x = x
        data[nt].x_mean = mean
        data[nt].x_std = std
    data = pull_timestamps(data)
    return data

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

def load_and_preprocess_data(path='data3.pt'):
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
    data = compute_controversiality(clean_features(data))

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
    def __init__(self, hidden_dim, dropout, device=device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        self.process_map = {
            ('donor', 'donated_to', 'legislator_term'): 'donation',
            ('lobby_firm', 'lobbied', 'legislator_term'): 'lobbying',
            ('bill_version', 'is_version', 'bill'): 'hierarchy'
        }

        self.convs = nn.ModuleDict({
            _et_to_key(et): SAGEConv((hidden_dim, hidden_dim), hidden_dim, normalize=True)
            for et in self.process_map.keys()
        })

        self.temporal_encoder = LegislativeTemporalEncoder(hidden_dim)
        self.vote_conv = PolarityAwareConv(hidden_dim, 385, dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, ts_dict):
        out = {nt: torch.zeros_like(x, device=x.device) for nt, x in x_dict.items()}

        for key, conv in self.convs.items():
            et = _key_to_et(key)
            src, rel, dst = et
            if et not in edge_index_dict:
                continue

            edge_index = edge_index_dict[et]
            if edge_index.numel() == 0:
                continue
            x_src = x_dict[src]
            if ts_dict is not None and et in ts_dict and ts_dict[et] is not None:
                edge_temp = self.temporal_encoder(ts_dict[et], self.process_map.get(et, 'vote'))
                if edge_index[0].max() < x_dict[src].size(0):
                    node_temp = scatter_mean(edge_temp, edge_index[0], dim=0, dim_size=x_dict[src].size(0))
                    x_src = x_src + node_temp

            if (edge_index[0].max() < x_src.size(0) and
                edge_index[1].max() < x_dict[dst].size(0)):
                conv_out = conv((x_src, x_dict[dst]), edge_index)
                out[dst] += conv_out

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
    def __init__(self, in_dims, cluster_id, topic_onehot, hidden_dim, dropout, device=device):
        super().__init__()
        self.device = device

        self.node_types = in_dims.keys()

        self.feature_proj = nn.ModuleDict({
            nt: nn.Sequential(
                    nn.LayerNorm(in_dims[nt]),
                    nn.Linear(in_dims[nt], hidden_dim, bias=False),
                    nn.GELU(),
                )
            for nt in in_dims
        })

        self.encoders = nn.ModuleList([LegislativeGraphEncoder(hidden_dim, dropout, device) for _ in range(n_layers)])

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
        self.loss_computer = StableLossComputer(device=device)

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
        try:
            return torch.clamp(total_loss / max(num_edges, 1), 0, 1000)
        except:
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

    model = LegislativeGraphModel(in_dims, cluster_bill, topic_onehot_bill, hidden_dim, dropout_p).to(device)
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