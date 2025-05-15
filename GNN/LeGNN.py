import datetime, gc, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import SAGEConv, GINEConv, LEConv
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.loader import HGTLoader

hidden_dim = 192
n_layers = 3
dropout_p = 0.10
device = torch.device('mps')

# utilities
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

# preprocessing
def safe_normalize_timestamps(timestamps):
    timestamps = torch.nan_to_num(timestamps, nan=0.0, posinf=1e4, neginf=-1e4)
    min_time = timestamps.min()
    max_time = timestamps.max()
    if (max_time - min_time) < 1e-4:
        return torch.zeros_like(timestamps)
    return (timestamps - min_time) / (max_time - min_time)

def safe_standardize_time_format(time_data):
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
            elif isinstance(t, datetime.datetime):
                td = t.timestamp()
            else:
                td = float(t) * 1e9
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

def load_and_preprocess_data(path='GNN/data2.pt'):
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

    data = ToUndirected(merge=False)(full_data)
    del full_data
    gc.collect()
    data = RemoveIsolatedNodes()(data)
    data = compute_controversiality(clean_features(data))

    for nt in data.node_types:
        ids = torch.arange(data[nt].num_nodes, device=device)
        data[nt].node_id = ids
    for store in data.stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                store[key] = value.float()
    return data

# encoders
class Time2Vec(nn.Module):
    def __init__(self, d=12):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(()))
        self.w = nn.Parameter(torch.randn(d - 1))
        self.b = nn.Parameter(torch.randn(d))
    def forward(self, t):
        v0 = self.w0 * t + self.b[0]
        v = torch.sin(self.w * t.unsqueeze(-1) + self.b[1:])
        return torch.cat([v0.unsqueeze(-1), v], -1)

class FeatureProjector(nn.Module):
    def __init__(self, metadata, in_dims, d_out=hidden_dim, t_dim=12):
        super().__init__()
        self.t2v = Time2Vec(t_dim)
        self.prj = nn.ModuleDict()
        for nt in metadata[0]:
            use_t = nt in ['bill_version','legislator_term', 'bill']
            d_in  = in_dims[nt] + (t_dim if use_t else 0)
            self.prj[nt] = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_out, bias=False),
                nn.GELU())
            _init_linear(self.prj[nt][1])

    def forward(self, x, ts):
        out={}
        for nt,xn in x.items():
            if ts and ts[nt] is not None:
                xn = torch.cat([xn, self.t2v(ts[nt])], dim=1)
            out[nt]=self.prj[nt](xn)
        return out


class RGTLayer(nn.Module):
    def __init__(self, d=hidden_dim, h=4, p=0.1, device=device):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.d = d
        self.device = device
        self.Q = nn.Linear(self.d, self.d, bias=False)
        self.K = nn.Linear(self.d, self.d, bias=False)
        self.V = nn.Linear(self.d, self.d, bias=False)
        self.rel = nn.ParameterDict()
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4*d, bias=False),
            nn.GELU(),
            nn.Linear(4*d, d, bias=False),
            nn.Dropout(p))

    def _r(self,r):
        k='__'.join(r)
        if k not in self.rel:
            param = nn.Parameter(torch.randn(self.h, self.dk, device=self.device))
            self.register_parameter(k, param)
            self.rel[k] = param
        else:
            param = self.rel[k]
            if param.device != self.device:
                param.data = param.data.to(self.device)
        return param

    def forward(self, h, ei, ew):
        msg = {k: torch.zeros_like(v) for k, v in h.items()}
        for r, e in ei.items():
            s, t = e
            q = self.Q(h[r[2]])[t].view(-1, self.h, self.dk)
            k = self.K(h[r[0]])[s].view(-1, self.h, self.dk)
            v = self.V(h[r[0]])[s].view(-1, self.h, self.dk)
            R = self._r(r)
            l = ((q * (k + R)).sum(-1)) / np.sqrt(self.dk)
            a = softmax(F.leaky_relu(l), t, num_nodes=h[r[2]].size(0))
            if (w := ew.get(r)) is not None:
                a = a * (1. + w.unsqueeze(-1))
            m = (a.unsqueeze(-1) * v).view(-1, self.h * self.dk)
            msg[r[2]] += scatter_add(m, t, dim=0, dim_size=h[r[2]].size(0))
        o = {}
        for nt in h:
            h_res = h[nt] + msg[nt]
            o[nt] = h_res + self.ffn(F.layer_norm(h_res, (h_res.size(1),)))
        return o

class PolarityAwareConv(nn.Module):
    def __init__(self, h_dim, e_dim, p, inc, outc):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(e_dim - 1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p),
        )

    def forward(self, x, ei, ea):
        x_src, x_dst = x
        src, dst = ei

        pol = ea[:, :1].clamp(0, 1)
        raw = ea[:, 1:]
        if raw.dim() == 1:
            raw = raw.unsqueeze(-1)

        e_feat = self.edge_mlp(raw) * (pol + 0.01)
        src_emb = x_src[src]
        m = src_emb * e_feat

        return scatter_add(m, dst, dim=0, dim_size=x_dst.size(0))

class ActorTopicAligner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Q = nn.Linear(d, d, bias=False)
        self.K = nn.Linear(d, d, bias=False)

class BillTopicHead(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, k, bias=False)
    def forward(self, z):
        return self.fc(z)

class InfluenceScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Q = nn.Linear(d, d, bias=False)
        self.K = nn.Linear(d, d, bias=False)

class SuccessHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, 1)

    def forward(self, z):
        return self.l(z).squeeze(-1)

class HierarchyAggregator(nn.Module):
    def forward(self, z, b):
        s, d = b.edge_index_dict[('bill_version', 'is_version', 'bill')]
        bill_agg = scatter_mean(
            z['bill_version'][s],
            d,
            dim=0,
            dim_size=z['bill'].size(0)
        )
        z['bill'] = 0.7 * z['bill'] + 0.3 * bill_agg
        d2, s2 = b.edge_index_dict[('legislator', 'samePerson', 'legislator_term')]
        leg_agg = scatter_mean(
            z['legislator_term'][s2],
            d2,
            dim=0,
            dim_size=z['legislator'].size(0)
        )
        z['legislator'] = 0.7 * z['legislator'] + 0.3 * leg_agg
        return z

class LegislativeGraphEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim, n_layers, dropout, device=device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]

        in_dims = {nt: c for nt, c in zip(self.node_types, [389, 769, 385, 2, 385, 384, 384])}
        self.proj = FeatureProjector(metadata, in_dims, d_out=hidden_dim, t_dim=12)

        self.rgt_layers = nn.ModuleList([RGTLayer(hidden_dim, 4, dropout)
                                        for _ in range(n_layers)])

        vote_e_dim = 385
        self.vote_conv = PolarityAwareConv(hidden_dim, vote_e_dim,
                                            dropout, hidden_dim, hidden_dim)

        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim).to(device)
                                    for nt in self.node_types})
        self.drop = nn.Dropout(dropout)

    def forward(self, batch):
        ts_dict = {nt: getattr(batch[nt], 'timestamp', None) for nt in self.node_types}
        h = self.proj({nt: batch[nt].x.to(self.device) for nt in self.node_types}, ts_dict)

        edge_w = {et: getattr(batch[et], 'edge_weight', None) for et in batch.edge_types}
        for layer in self.rgt_layers:
            h = layer(h, batch.edge_index_dict, edge_w)

        vote_ei = batch.edge_index_dict[('legislator_term', 'voted_on', 'bill_version')]
        vote_attr = batch[('legislator_term', 'voted_on', 'bill_version')].edge_attr
        vote_msg = self.vote_conv(
            (h["legislator_term"], h["bill_version"]),
            vote_ei,
            vote_attr,
        )
        h["bill_version"] += vote_msg

        h = {nt: self.drop(F.relu(self.norm[nt](h[nt]))) for nt in h}
        return h

class LegislativeGraphModel(nn.Module):
    def __init__(self, metadata, cluster_id, topic_onehot, hidden_dim, dropout, device=device):
        super().__init__()
        self.device = device
        self.encoder = LegislativeGraphEncoder(metadata, hidden_dim, n_layers, dropout, device)
        self.hier = HierarchyAggregator()
        self.topic_head = BillTopicHead(hidden_dim, topic_onehot.size(1))
        self.success = SuccessHead(hidden_dim)

        self.actor_types = ['legislator', 'committee', 'donor', 'lobby_firm']
        self.align = nn.ModuleDict({nt: ActorTopicAligner(hidden_dim) for nt in self.actor_types})
        self.inf = nn.ModuleDict({nt: InfluenceScorer(hidden_dim) for nt in self.actor_types})

        self.register_buffer('cluster_id',  cluster_id)
        self.register_buffer('topic_onehot', topic_onehot)

    def forward(self, batch, mask_weight_dict):
        z = self.hier(self.encoder(batch), batch)
        bill_nodes = batch['bill'].node_id.to(self.device)
        bv_nid = batch['bill_version'].node_id.to(self.device)
        z_bv = z['bill_version'][bv_nid]
        z_actor = {nt: z[nt] for nt in self.actor_types if nt in z}

        bill_embed = z['bill'][bill_nodes]
        bill_logits = self.topic_head(bill_embed)
        bill_labels = self.cluster_id[bill_nodes]
        bill_prob = self.topic_onehot[bill_nodes]
        success_log = self.success(bill_embed)[bill_nodes]

        ev2b = batch.edge_index_dict[('bill_version', 'is_version', 'bill')]
        av2b = batch[('bill_version', 'is_version', 'bill')].edge_attr[:, 0].clamp(1e-2, 1e2).to(self.device)
        B_vb = bv_nid.size(0)
        Bb = bill_nodes.size(0)
        W_vb = torch.zeros(B_vb, Bb, device=self.device)
        rows, cols = ev2b
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        W_vb[rows, bill_nodes[cols]] = av2b

        align_dict, infl_dict = {}, {}
        for nt, az in z_actor.items():
            mvb, wvb = mask_weight_dict[nt]
            mvb = mvb.to(self.device, non_blocking=True)
            wvb = wvb.to(self.device, non_blocking=True)

            q_t = F.normalize(self.align[nt].Q(az))
            k_t = F.normalize(self.align[nt].K(z_bv))
            sim_t = (q_t @ k_t.T) / math.sqrt(az.size(-1))
            sim_t = sim_t.masked_fill(~mvb, -1e9)
            if wvb is not None:
                sim_t = sim_t + wvb
            Pav = sim_t.softmax(1)
            Pab = Pav @ W_vb
            Pab = Pab / Pab.sum(1, keepdim=True).clamp(min=1e-9)
            align_dict[nt] = Pab @ bill_prob

            q_i = F.normalize(self.inf[nt].Q(az))
            k_i = F.normalize(self.inf[nt].K(z_bv))
            sim_i = (q_i @ k_i.T) / math.sqrt(az.size(-1))
            sim_i = sim_i.masked_fill(~mvb, -1e9)
            if wvb is not None:
                sim_i = sim_i + wvb
            Pav_i = sim_i.softmax(1)
            Pab_i = Pav_i @ W_vb
            infl_dict[nt] = (
                Pab_i * torch.sigmoid(success_log).unsqueeze(0)
            ).sum(dim=1)

        return {
            "node_embeddings": z,
            "bill_logits": bill_logits,
            "bill_labels": bill_labels,
            "success_logit": success_log,
            "actor_topic_dict": align_dict,
            "influence_dict" : infl_dict
        }

    def compute_loss(self, out, batch, neg_k=1, temp=0.1):
        ce = F.cross_entropy(out["bill_logits"], out["bill_labels"])
        if torch.isnan(ce) or torch.isinf(ce):
            ce = torch.tensor(0.0, device=self.device)
        link = 0.0
        bce = nn.BCEWithLogitsLoss()
        z = out['node_embeddings']
        for (s_t, _, d_t), ei in batch.edge_index_dict.items():
            s, d = ei
            pos = (F.normalize(z[s_t][s], -1) * F.normalize(z[d_t][d], -1)).sum(-1) / temp
            link += bce(pos, torch.ones_like(pos))
            if neg_k:
                perm = d[torch.randperm(d.size(0), device=self.device)][: pos.size(0) * neg_k]
                neg = (
                    F.normalize(z[s_t][s].repeat(neg_k, 1), -1)
                    * F.normalize(z[d_t][perm], -1)
                ).sum(-1) / temp
                link += bce(neg, torch.zeros_like(neg))

        infl_vals = []
        for v in out['influence_dict'].values():
            v = v[torch.isfinite(v)]
            if v.numel() > 0:
                infl_vals.append(v.pow(2).mean())
        inf_reg = torch.stack(infl_vals).mean() if infl_vals else torch.tensor(0., device=self.device)

        ent_values = []
        for probs in out['actor_topic_dict'].values():
            p = torch.nan_to_num(probs, nan=0.0, posinf=1e8, neginf=-1e2)
            if p.sum() == 0:
                continue
            row_sum = p.sum(dim=1, keepdim=True)
            mask = (row_sum.squeeze(-1) > 0)
            p_norm = torch.zeros_like(p)
            p_norm[mask] = p[mask] / row_sum[mask]
            entropy = (-p_norm[mask] * (p_norm[mask] + 1e-9).log()).sum(dim=1)
            if entropy.numel() > 0:
                ent_values.append(entropy.mean())
        ent_reg = torch.stack(ent_values).mean() if ent_values else torch.tensor(0., device=self.device)

        total = ce + link + 0.1 * inf_reg + 0.1 * ent_reg
        return {
            'total_loss': total,
            'ce_loss': ce,
            'link_loss': link,
            'inf_reg': inf_reg,
            'ent_reg': ent_reg
        }
def _adj(data, et):
    row, col = data.edge_index_dict[et]
    ns, nd = data[et[0]].num_nodes, data[et[2]].num_nodes
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.ones_like(row, dtype=torch.float32), (ns, nd)
    ).coalesce()

def build_masks_weights(d, acts, A_by, BV2B, paths, device=device):
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

def main():
    with open('node_id_map.json', 'r') as f:
        node_id_map = json.load(f)

    with open('bill_labels.json', 'r') as f:
        topic_cluster_labels_dict = json.load(f)

    data = load_and_preprocess_data()
    bv2bill = data.edge_index_dict[("bill_version", "is_version", "bill")]
    parent_bill = torch.full((data["bill_version"].num_nodes,), -1, dtype=torch.long)
    parent_bill[bv2bill[0]] = bv2bill[1]

    num_topics = max(topic_cluster_labels_dict.values()) + 1
    cluster_bv = torch.full_like(parent_bill, -1)
    for bill_id, bill_node in node_id_map['bill_version'].items():
        cluster_bv[parent_bill == bill_node] = topic_cluster_labels_dict.get(bill_id, num_topics)

    topic_onehot_bv = F.one_hot(cluster_bv.clamp(min=0), num_classes=num_topics+1).float()
    cluster_bv, topic_onehot_bv = cluster_bv.to(device), topic_onehot_bv.to(device)
    A_by = {et: _adj(data, et).coalesce() for et in data.edge_types}
    BV2B = _adj(data, ('bill_version', 'is_version', 'bill')).coalesce()

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

    best_loss = float('inf')
    model = LegislativeGraphModel(data.metadata(), cluster_bv, topic_onehot_bv, hidden_dim, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    torch.mps.empty_cache()
    gc.collect()
    loader = HGTLoader(
        data,
        num_samples={
            'legislator_term': [48] * 2,
            'legislator': [48] * 2,
            'committee': [96] * 2,
            'lobby_firm': [192] * 2,
            'donor': [192] * 2,
            'bill_version': [384] * 2,
            'bill': [384] * 2,
        },
        batch_size=64,
        shuffle=True,
        input_nodes='legislator_term'
    )

    epochs = 1
    patience = 5
    counter = 0
    start_epoch = 0
    best_loss = float('inf')
    # model.load_state_dict(torch.load("best_model5.pt"))
    # optimizer.load_state_dict(torch.load("best_opt5.pt"))

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), position=0, total=epochs):
        model.train()
        epoch_loss = 0.0
        torch.mps.empty_cache()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", position=1)):
            batch = batch.to(device)
            mw_dict = build_masks_weights(batch, ['legislator','committee','donor','lobby_firm'], A_by, BV2B, ALLOWED)
            outputs = model(batch, mw_dict)
            loss_dict = model.compute_loss(outputs, batch)
            loss = loss_dict['total_loss'].float()

            if torch.isnan(loss) or torch.isinf(loss):
                print(loss_dict)
                raise ValueError(f"Batch {batch_idx}: NaN/Inf loss detected!")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            del outputs, loss_dict, loss
            torch.mps.empty_cache()
            gc.collect()


        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "best_model5.pt")
            torch.save(optimizer.state_dict(), "best_opt5.pt")
            print(f"  New best loss; resetting patience counter.")
        else:
            counter += 1
            print(f"  No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()