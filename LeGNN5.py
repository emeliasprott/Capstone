import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_geometric.nn import SAGEConv, HeteroConv


class CFG:
    hidden_dim = 148
    num_layers = 3
    dropout = 0.15
    lr = 3e-4
    weight_decay = 1e-4
    epochs = 5

    num_topics = 66
    actor_types = ["legislator_term", "committee", "donor", "lobby_firm"]
    input_type = "bill"

    num_neighbors = {
        ("bill_version", "is_version", "bill"): [2, 1, 0],
        ("bill_version", "priorVersion", "bill_version"): [4, 2, 1],
        ("legislator", "samePerson", "legislator_term"): [4, 2, 1],
        ("legislator_term", "wrote", "bill_version"): [16, 8, 4],
        ("legislator_term", "member_of", "committee"): [8, 4, 4],
        ("lobby_firm", "lobbied", "legislator_term"): [4, 4, 2],
        ("lobby_firm", "lobbied", "committee"): [4, 4, 2],
        ("donor", "donated_to", "legislator_term"): [4, 4, 2],
        ("legislator_term", "voted_on", "bill_version"): [32, 16, 4],
        ("bill_version", "read", "committee"): [16, 8, 4],
    }

    batch_size = 3048
    num_workers = 0

    lambda_bill = 1.0
    lambda_vote = 1.0
    lambda_donation = 0.2
    lambda_lobby = 0.2

    lambda_stance_reg = 1e-2
    lambda_infl_reg = 5e-3

    lambda_vote_orient = 2e-4
    lambda_money_orient = 5e-4

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Label prep ----------


def pad_bill_outcomes(data):
    if "bill" not in data.node_types:
        raise ValueError("Missing 'bill' node type.")
    if not hasattr(data["bill"], "y"):
        raise ValueError("data['bill'].y missing.")

    y_raw = data["bill"].y.view(-1).long()
    n = data["bill"].num_nodes

    if y_raw.size(0) < n:
        pad = torch.zeros(n - y_raw.size(0), dtype=torch.long)
        y = torch.cat([y_raw, pad], dim=0)
    else:
        y = y_raw[:n]

    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[: min(n, y_raw.size(0))] = True

    data["bill"].y = y
    data["bill"].y_mask = mask


def ensure_bill_topics(data, num_topics):
    if not hasattr(data["bill"], "cluster"):
        raise ValueError("data['bill'].cluster missing.")
    t = data["bill"].cluster.view(-1).long()
    n = data["bill"].num_nodes
    if t.size(0) != n:
        raise ValueError("bill.cluster length mismatch.")
    valid = t[t >= 0]
    if valid.numel() > 0 and (valid.min() < 0 or valid.max() >= num_topics):
        raise ValueError("bill.cluster out of range.")
    data["bill"].topic_id = t
    data["bill"].has_topic = t >= 0


def attach_bill_version_labels(data, num_topics):
    pad_bill_outcomes(data)
    ensure_bill_topics(data, num_topics)

    et = ("bill_version", "is_version", "bill")
    if et not in data.edge_types:
        raise ValueError("Missing ('bill_version','is_version','bill') edges.")

    rel = data[et]
    bv = rel.edge_index[0]
    b = rel.edge_index[1]

    num_bv = data["bill_version"].num_nodes
    outcome_y = torch.zeros(num_bv, dtype=torch.long)
    outcome_mask = torch.zeros(num_bv, dtype=torch.bool)
    topic_id = torch.full((num_bv,), -1, dtype=torch.long)
    has_topic = torch.zeros(num_bv, dtype=torch.bool)

    by = data["bill"].y
    by_mask = data["bill"].y_mask
    bt = data["bill"].topic_id
    b_has = data["bill"].has_topic

    outcome_y[bv] = by[b]
    outcome_mask[bv] = by_mask[b]
    topic_id[bv] = bt[b]
    has_topic[bv] = b_has[b]

    data["bill_version"].outcome_y = outcome_y
    data["bill_version"].outcome_mask = outcome_mask
    data["bill_version"].topic_id = topic_id
    data["bill_version"].has_topic = has_topic


# ---------- Encoders ----------


class NodeEncoder(nn.Module):
    def __init__(self, data, hidden_dim, dropout):
        super().__init__()
        self.lins = nn.ModuleDict()
        self.embeds = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        for nt in data.node_types:
            if hasattr(data[nt], "x"):
                in_dim = data[nt].x.size(1)
                self.lins[nt] = nn.Linear(in_dim, hidden_dim)
                self.norms[nt] = nn.LayerNorm(hidden_dim)
            else:
                self.embeds[nt] = nn.Embedding(data[nt].num_nodes, hidden_dim)
                self.norms[nt] = nn.LayerNorm(hidden_dim)

    def forward(self, batch):
        h = {}
        for nt in batch.node_types:
            if nt in self.lins and hasattr(batch[nt], "x"):
                x = batch[nt].x
                h_nt = self.lins[nt](x)
            elif nt in self.embeds:
                if hasattr(batch[nt], "n_id"):
                    idx = batch[nt].n_id
                else:
                    idx = torch.arange(
                        batch[nt].num_nodes,
                        device=self.embeds[nt].weight.device,
                    )
                h_nt = self.embeds[nt](idx)
            else:
                continue

            h_nt = self.norms[nt](h_nt)
            h[nt] = self.dropout(F.relu(h_nt))

        return h


class HeteroSAGE(nn.Module):
    def __init__(self, data, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        node_types, edge_types = data.metadata()

        for _ in range(num_layers):
            conv_dict = {}
            for src, rel, dst in edge_types:
                conv_dict[(src, rel, dst)] = SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    aggr="mean",
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
            )

    def forward(self, x_dict, edge_index_dict):
        h = x_dict
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index_dict)
            for nt in h:
                h_nt = self.norms[layer][nt](h[nt])
                h[nt] = self.dropout(F.relu(h_nt))
        return h


# ---------- Latent modules ----------


class TopicStanceModule(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_topics,
        actor_types,
        tau=3.0,
        target_norm=0.3,
        target_abs=0.3,
    ):
        super().__init__()
        self.actor_types = actor_types
        self.tau = tau
        self.target_norm = target_norm
        self.target_abs = target_abs
        self.proj = nn.ModuleDict(
            {at: nn.Linear(hidden_dim, num_topics) for at in actor_types}
        )

    def forward(self, h_dict):
        out = {}
        for at in self.actor_types:
            if at in h_dict:
                logits = self.proj[at](h_dict[at])
                out[at] = torch.tanh(logits / self.tau)
        return out

    def regularization(self, stance_dict):
        total = 0.0
        count = 0
        for S in stance_dict.values():
            if S.numel() == 0:
                continue
            mean_per_topic = S.mean(dim=0)
            total = total + (mean_per_topic**2).mean()
            l2 = S.pow(2).mean(dim=1)
            total = total + ((l2 - self.target_norm) ** 2).mean()
            mean_abs = S.abs().mean()
            total = total + ((mean_abs - self.target_abs) ** 2)
            count += 1
        if count == 0:
            return None
        return total / count


class InfluenceModule(nn.Module):
    def __init__(self, hidden_dim, actor_types):
        super().__init__()
        self.actor_types = actor_types
        self.mlp = nn.ModuleDict()
        for at in actor_types:
            self.mlp[at] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, h_dict):
        out = {}
        for at in self.actor_types:
            if at in h_dict:
                v = self.mlp[at](h_dict[at]).squeeze(-1)
                out[at] = F.softplus(v)
        return out

    def regularization(self, infl_dict):
        vals = []
        for v in infl_dict.values():
            if v.numel() > 0:
                vals.append(torch.log1p(v))
        if not vals:
            return None
        all_v = torch.cat(vals, dim=0)
        mean = all_v.mean()
        std = all_v.std()
        if std < 1e-6:
            std = all_v.new_tensor(1.0)
        reg_mean = mean.pow(2)
        reg_std = (std - 1.0).pow(2)
        return reg_mean + reg_std


# ---------- Task heads ----------


class BillOutcomeHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, h_bill):
        return self.mlp(h_bill)


class VoteHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, lt_h, bv_h, s_topic, i_lt):
        x = torch.cat(
            [
                lt_h,
                bv_h,
                s_topic.unsqueeze(-1),
                i_lt.unsqueeze(-1),
                (s_topic * i_lt).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(x)


class EdgeAmountHead(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.mlp(features).squeeze(-1)


# ---------- Full model ----------


class LegModel(nn.Module):
    def __init__(self, data, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.node_encoder = NodeEncoder(data, cfg.hidden_dim, cfg.dropout)
        self.gnn = HeteroSAGE(
            data,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.dropout,
        )
        self.topic_module = TopicStanceModule(
            cfg.hidden_dim,
            cfg.num_topics,
            cfg.actor_types,
            tau=3.0,
            target_norm=0.3,
            target_abs=0.3,
        )
        self.infl_module = InfluenceModule(cfg.hidden_dim, cfg.actor_types)
        self.bill_head = BillOutcomeHead(cfg.hidden_dim)
        self.vote_head = VoteHead(cfg.hidden_dim)
        self.edge_head = EdgeAmountHead(in_dim=3)

    def forward(self, batch):
        x = self.node_encoder(batch)
        h = self.gnn(x, batch.edge_index_dict)
        stance = self.topic_module(h)
        infl = self.infl_module(h)
        return h, stance, infl


def zero_loss_like(model):
    p = next(model.parameters())
    return p.new_zeros(())


# ---------- Losses ----------


def loss_bill_outcome(model, h, batch, cfg):
    if "bill" not in batch.node_types:
        return zero_loss_like(model)
    node = batch["bill"]
    if not hasattr(node, "y") or not hasattr(node, "y_mask"):
        return zero_loss_like(model)

    y = node.y.view(-1)
    m = node.y_mask.view(-1)
    if m.sum() == 0 or "bill" not in h:
        return zero_loss_like(model)

    logits = model.bill_head(h["bill"])
    y3 = (y[m] + 1).clamp(0, 2)
    return F.cross_entropy(logits[m], y3)


def loss_vote_ce(model, h, stance, infl, batch, cfg):
    et = ("legislator_term", "voted_on", "bill_version")
    if et not in batch.edge_types:
        return zero_loss_like(model)
    rel = batch[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        return zero_loss_like(model)

    if "legislator_term" not in h or "bill_version" not in h:
        return zero_loss_like(model)
    if "legislator_term" not in stance or "legislator_term" not in infl:
        return zero_loss_like(model)
    if not (
        hasattr(batch["bill_version"], "topic_id")
        and hasattr(batch["bill_version"], "has_topic")
    ):
        return zero_loss_like(model)

    lt = rel.edge_index[0]
    bv = rel.edge_index[1]
    vote = rel.edge_attr[:, -1].long()
    tb = batch["bill_version"].topic_id[bv]
    has_t = batch["bill_version"].has_topic[bv]

    m = (vote != 0) & has_t
    if m.sum() == 0:
        return zero_loss_like(model)

    lt = lt[m]
    bv = bv[m]
    v = vote[m]
    topics = tb[m].clamp(min=0, max=cfg.num_topics - 1)

    lt_h = h["legislator_term"][lt]
    bv_h = h["bill_version"][bv]

    S_lt = stance["legislator_term"][lt]
    s_topic = S_lt[torch.arange(S_lt.size(0), device=S_lt.device), topics]
    i_lt = infl["legislator_term"][lt]

    logits = model.vote_head(lt_h, bv_h, s_topic, i_lt)
    y3 = (v + 1).clamp(0, 2)
    return F.cross_entropy(logits, y3)


def loss_vote_orient(stance, batch, cfg):
    et = ("legislator_term", "voted_on", "bill_version")
    if et not in batch.edge_types:
        return None
    if "legislator_term" not in stance:
        return None
    if not (
        hasattr(batch["bill_version"], "topic_id")
        and hasattr(batch["bill_version"], "has_topic")
    ):
        return None

    rel = batch[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        return None

    lt = rel.edge_index[0]
    bv = rel.edge_index[1]
    v = rel.edge_attr[:, -1].float()

    tb = batch["bill_version"].topic_id[bv]
    has_t = batch["bill_version"].has_topic[bv]

    m = (v != 0) & has_t
    if m.sum() == 0:
        return None

    lt = lt[m]
    v = v[m]
    topics = tb[m].clamp(min=0, max=cfg.num_topics - 1)

    S_lt = stance["legislator_term"]
    s_topic = S_lt[lt, topics]

    margin = 0.05
    prod = v * s_topic
    viol = F.relu(margin - prod)
    return viol.mean()


def normalize_log_amount(amount):
    log_amt = torch.log1p(torch.clamp(amount, min=0.0))
    if log_amt.numel() == 0:
        return log_amt
    mean = log_amt.mean()
    std = log_amt.std()
    if std < 1e-6:
        std = torch.tensor(1.0, device=log_amt.device)
    return (log_amt - mean) / std


def loss_donation(model, h, stance, infl, batch, cfg):
    et = ("donor", "donated_to", "legislator_term")
    if et not in batch.edge_types:
        return zero_loss_like(model)
    if "donor" not in stance or "legislator_term" not in stance:
        return zero_loss_like(model)

    rel = batch[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        return zero_loss_like(model)

    d = rel.edge_index[0]
    lt = rel.edge_index[1]
    amount = rel.edge_attr[:, 0].float()
    m = amount > 0
    if m.sum() == 0:
        return zero_loss_like(model)

    d = d[m]
    lt = lt[m]
    amount = amount[m]

    S_d = stance["donor"][d]
    S_lt = stance["legislator_term"][lt]
    sim = (S_d * S_lt).mean(dim=-1)

    i_d = infl.get("donor")
    i_lt = infl.get("legislator_term")
    if i_d is not None and i_lt is not None:
        i_feat = i_d[d] + i_lt[lt]
    else:
        i_feat = sim.new_zeros(sim.size(0))

    feat = torch.stack([sim, i_feat, sim * i_feat], dim=-1)
    target = normalize_log_amount(amount)
    if target.numel() == 0:
        return zero_loss_like(model)

    pred = model.edge_head(feat)
    return F.mse_loss(pred, target)


def loss_donation_orient(stance, batch, cfg):
    et = ("donor", "donated_to", "legislator_term")
    if et not in batch.edge_types:
        return None
    if "donor" not in stance or "legislator_term" not in stance:
        return None

    rel = batch[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        return None

    d = rel.edge_index[0]
    lt = rel.edge_index[1]
    amt = rel.edge_attr[:, 0].float()

    m = amt > 0
    if m.sum() == 0:
        return None

    d = d[m]
    lt = lt[m]

    S_d = stance["donor"][d]
    S_lt = stance["legislator_term"][lt]
    sim = (S_d * S_lt).mean(dim=-1)

    viol = F.relu(-sim)
    return viol.mean()


def loss_lobby(model, h, stance, infl, batch, cfg):
    total = None
    count = 0

    if "lobby_firm" in stance:
        if (
            "lobby_firm",
            "lobbied",
            "legislator_term",
        ) in batch.edge_types and "legislator_term" in stance:
            rel = batch[("lobby_firm", "lobbied", "legislator_term")]
            if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
                lf = rel.edge_index[0]
                lt = rel.edge_index[1]
                amount = rel.edge_attr[:, 0].float()
                m = amount > 0
                if m.sum() > 0:
                    lf = lf[m]
                    lt = lt[m]
                    amount = amount[m]
                    S_lf = stance["lobby_firm"][lf]
                    S_lt = stance["legislator_term"][lt]
                    sim = (S_lf * S_lt).mean(dim=-1)
                    i_lf = infl.get("lobby_firm")
                    i_lt = infl.get("legislator_term")
                    if i_lf is not None and i_lt is not None:
                        i_feat = i_lf[lf] + i_lt[lt]
                    else:
                        i_feat = sim.new_zeros(sim.size(0))
                    feat = torch.stack([sim, i_feat, sim * i_feat], dim=-1)
                    target = normalize_log_amount(amount)
                    if target.numel() > 0:
                        pred = model.edge_head(feat)
                        val = F.mse_loss(pred, target)
                        total = val if total is None else total + val
                        count += 1

        if (
            "lobby_firm",
            "lobbied",
            "committee",
        ) in batch.edge_types and "committee" in stance:
            rel = batch[("lobby_firm", "lobbied", "committee")]
            if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
                lf = rel.edge_index[0]
                cm = rel.edge_index[1]
                amount = rel.edge_attr[:, 0].float()
                m = amount > 0
                if m.sum() > 0:
                    lf = lf[m]
                    cm = cm[m]
                    amount = amount[m]
                    S_lf = stance["lobby_firm"][lf]
                    S_cm = stance["committee"][cm]
                    sim = (S_lf * S_cm).mean(dim=-1)
                    i_lf = infl.get("lobby_firm")
                    i_cm = infl.get("committee")
                    if i_lf is not None and i_cm is not None:
                        i_feat = i_lf[lf] + i_cm[cm]
                    else:
                        i_feat = sim.new_zeros(sim.size(0))
                    feat = torch.stack([sim, i_feat, sim * i_feat], dim=-1)
                    target = normalize_log_amount(amount)
                    if target.numel() > 0:
                        pred = model.edge_head(feat)
                        val = F.mse_loss(pred, target)
                        total = val if total is not None else val
                        count += 1

    if count == 0:
        return zero_loss_like(model)
    return total / count


def loss_lobby_orient(stance, batch, cfg):
    if "lobby_firm" not in stance:
        return None

    losses = []

    if (
        "lobby_firm",
        "lobbied",
        "legislator_term",
    ) in batch.edge_types and "legislator_term" in stance:
        rel = batch[("lobby_firm", "lobbied", "legislator_term")]
        if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
            lf = rel.edge_index[0]
            lt = rel.edge_index[1]
            amt = rel.edge_attr[:, 0].float()
            m = amt > 0
            if m.sum() > 0:
                lf = lf[m]
                lt = lt[m]
                S_lf = stance["lobby_firm"][lf]
                S_lt = stance["legislator_term"][lt]
                sim = (S_lf * S_lt).mean(dim=-1)
                losses.append(F.relu(-sim).mean())

    if (
        "lobby_firm",
        "lobbied",
        "committee",
    ) in batch.edge_types and "committee" in stance:
        rel = batch[("lobby_firm", "lobbied", "committee")]
        if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
            lf = rel.edge_index[0]
            cm = rel.edge_index[1]
            amt = rel.edge_attr[:, 0].float()
            m = amt > 0
            if m.sum() > 0:
                lf = lf[m]
                cm = cm[m]
                S_lf = stance["lobby_firm"][lf]
                S_cm = stance["committee"][cm]
                sim = (S_lf * S_cm).mean(dim=-1)
                losses.append(F.relu(-sim).mean())

    if not losses:
        return None
    return sum(losses) / len(losses)


# ---------- Sampling / training / inference ----------


def resolve_num_neighbors(data, cfg):
    base = dict(cfg.num_neighbors)
    for et in data.edge_types:
        if et in base:
            continue
        rev = (et[2], et[1], et[0])
        if rev in base:
            v = base[rev]
            base[et] = [max(1, v[0] // 2)] * len(v)
        else:
            base[et] = [2, 2, 2]
    return base


def make_loader(data, cfg):
    if cfg.input_type not in data.node_types:
        raise ValueError(f"Input node type {cfg.input_type} not found in data.")
    n_neighbors = resolve_num_neighbors(data, cfg)
    input_nodes = (cfg.input_type, torch.arange(data[cfg.input_type].num_nodes))
    return NeighborLoader(
        data,
        num_neighbors=n_neighbors,
        input_nodes=input_nodes,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )


def train_epoch(model, loader, optimizer, cfg):
    model.train()
    device = cfg.device
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        h, stance, infl = model(batch)

        l_bill = loss_bill_outcome(model, h, batch, cfg)
        l_vote = loss_vote_ce(model, h, stance, infl, batch, cfg)
        l_don = loss_donation(model, h, stance, infl, batch, cfg)
        l_lobby = loss_lobby(model, h, stance, infl, batch, cfg)

        stance_reg = model.topic_module.regularization(stance) if stance else None
        infl_reg = model.infl_module.regularization(infl) if infl else None
        vote_orient = loss_vote_orient(stance, batch, cfg)
        don_orient = loss_donation_orient(stance, batch, cfg)
        lob_orient = loss_lobby_orient(stance, batch, cfg)

        loss = (
            cfg.lambda_bill * l_bill
            + cfg.lambda_vote * l_vote
            + cfg.lambda_donation * l_don
            + cfg.lambda_lobby * l_lobby
        )

        if stance_reg is not None:
            loss = loss + cfg.lambda_stance_reg * stance_reg
        if infl_reg is not None:
            loss = loss + cfg.lambda_infl_reg * infl_reg
        if vote_orient is not None:
            loss = loss + cfg.lambda_vote_orient * vote_orient
        if don_orient is not None:
            loss = loss + cfg.lambda_money_orient * don_orient
        if lob_orient is not None:
            loss = loss + cfg.lambda_money_orient * lob_orient

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

        del batch, h, stance, infl
        if device == "mps":
            torch.mps.empty_cache()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def infer_all(model, data, cfg, batch_size=None):
    model.eval()
    device = cfg.device
    model.to(device)
    n_neighbors = resolve_num_neighbors(data, cfg)
    if batch_size is None:
        batch_size = min(cfg.batch_size, 4096)

    embs, topic_pred, inf_pred, topic_infl = {}, {}, {}, {}

    for nt in data.node_types:
        num_nodes = data[nt].num_nodes
        if num_nodes == 0:
            continue

        loader = NeighborLoader(
            data,
            num_neighbors=n_neighbors,
            input_nodes=(nt, torch.arange(num_nodes)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        out = torch.zeros((num_nodes, cfg.hidden_dim), dtype=torch.float32)

        for batch in loader:
            batch = batch.to(device)
            h, stance, infl = model(batch)

            if nt not in batch.node_types or nt not in h:
                if device == "mps":
                    torch.mps.empty_cache()
                continue

            seed_n_id = batch[nt].n_id[: batch[nt].batch_size]
            seed_h = h[nt][: batch[nt].batch_size].detach().cpu()
            out[seed_n_id] = seed_h

            del batch, h, stance, infl, seed_h, seed_n_id
            if device == "mps":
                torch.mps.empty_cache()

        embs[nt] = out

    for at in cfg.actor_types:
        if at in embs:
            h_at = embs[at].to(device)
            s = model.topic_module({at: h_at})
            i = model.infl_module({at: h_at})
            if at in s:
                topic_pred[at] = s[at].detach().cpu()
            if at in i:
                inf_pred[at] = i[at].detach().cpu()
            if at in s and at in i:
                ti = i[at].unsqueeze(-1) * s[at]
                topic_infl[at] = ti.detach().cpu()

    model.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()

    return topic_pred, inf_pred, topic_infl


# ---------- Diagnostics ----------


@torch.no_grad()
def _safe_corr(x, y):
    x = x.view(-1).float()
    y = y.view(-1).float()
    m = torch.isfinite(x) & torch.isfinite(y)
    x = x[m]
    y = y[m]
    if x.numel() < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    xs = x.std()
    ys = y.std()
    if xs < 1e-8 or ys < 1e-8:
        return float("nan")
    x = x / xs
    y = y / ys
    return float((x * y).mean().item())


@torch.no_grad()
def _inspect_basic(topic_pred, inf_pred, topic_infl, actor_types):
    print("\n=== BASIC DISTRIBUTIONS ===")
    for at in actor_types:
        if at in inf_pred:
            p = inf_pred[at]
            print(f"\n[{at}] influence:")
            print("  shape:", tuple(p.shape))
            print("  min/max:", float(p.min()), float(p.max()))
            print("  mean/std:", float(p.mean()), float(p.std()))

        if at in topic_pred:
            T = topic_pred[at]
            print(f"\n[{at}] topic stance:")
            print("  shape:", tuple(T.shape))
            print("  mean(|stance|):", float(T.abs().mean()))
            print("  avg per-actor std:", float(T.std(dim=1).mean()))
            print("  frac(|stance|>0.5):", float((T.abs() > 0.5).float().mean()))

        if at in topic_infl:
            TI = topic_infl[at]
            print(f"\n[{at}] topic influence (i * stance):")
            print("  shape:", tuple(TI.shape))
            print("  mean:", float(TI.mean()))
            print("  std:", float(TI.std()))
            pos_frac = float((TI > 0).float().mean())
            neg_frac = float((TI < 0).float().mean())
            print("  frac(pos):", pos_frac, " frac(neg):", neg_frac)


@torch.no_grad()
def _check_non_collapse(topic_pred, inf_pred, topic_infl, actor_types):
    print("\n=== NON-COLLAPSE CHECK ===")
    for at in actor_types:
        if at in inf_pred:
            p = inf_pred[at]
            uq = torch.unique(p.round(decimals=4))
            print(f"[{at}] influence unique (rounded): {uq.numel()} values")

        if at in topic_pred:
            T = topic_pred[at]
            uq_rows = torch.unique(T.round(decimals=3), dim=0)
            print(f"[{at}] stance unique rows (rounded): {uq_rows.size(0)}")

        if at in topic_infl:
            TI = topic_infl[at]
            uq_rows = torch.unique(TI.round(decimals=3), dim=0)
            print(f"[{at}] topic influence unique rows (rounded): {uq_rows.size(0)}")


@torch.no_grad()
def _vote_alignment(data, topic_pred, inf_pred, num_topics):
    print("\n=== VOTE ALIGNMENT CHECK ===")
    if "legislator_term" not in topic_pred or "legislator_term" not in inf_pred:
        print("missing legislator_term predictions, skip.")
        return

    et = ("legislator_term", "voted_on", "bill_version")
    if et not in data.edge_types:
        print("missing voted_on edges, skip.")
        return

    if not (
        hasattr(data["bill_version"], "topic_id")
        and hasattr(data["bill_version"], "has_topic")
    ):
        print("missing bill_version topic labels, skip.")
        return

    rel = data[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        print("no vote edge_attr, skip.")
        return

    lt = rel.edge_index[0]
    bv = rel.edge_index[1]
    v = rel.edge_attr[:, -1].float()

    tb = data["bill_version"].topic_id[bv]
    has_t = data["bill_version"].has_topic[bv]

    m = (v != 0) & has_t
    if m.sum() == 0:
        print("no usable votes, skip.")
        return

    lt = lt[m]
    v = v[m]
    topics = tb[m].clamp(min=0, max=num_topics - 1)

    S_lt = topic_pred["legislator_term"]
    I_lt = inf_pred["legislator_term"]

    s_vals = S_lt[lt, topics]
    i_vals = I_lt[lt]
    vote_sign = torch.sign(v)

    align_score = i_vals * s_vals
    corr = _safe_corr(align_score, vote_sign)

    yes_mean = (
        float(align_score[vote_sign > 0].mean())
        if (vote_sign > 0).any()
        else float("nan")
    )
    no_mean = (
        float(align_score[vote_sign < 0].mean())
        if (vote_sign < 0).any()
        else float("nan")
    )

    print("edges used:", int(m.sum().item()))
    print("Pearson(align_score, vote_sign):", corr)
    print("mean align_score | YES:", yes_mean)
    print("mean align_score | NO:", no_mean)


@torch.no_grad()
def _donation_alignment(data, topic_pred):
    print("\n=== DONATION ALIGNMENT CHECK ===")
    if "donor" not in topic_pred or "legislator_term" not in topic_pred:
        print("missing donor or legislator_term stance, skip.")
        return

    et = ("donor", "donated_to", "legislator_term")
    if et not in data.edge_types:
        print("no donated_to edges, skip.")
        return
    rel = data[et]
    if rel.edge_attr is None or rel.edge_attr.size(0) == 0:
        print("no donation edge_attr, skip.")
        return

    d = rel.edge_index[0]
    lt = rel.edge_index[1]
    amt = rel.edge_attr[:, 0].float()

    m = amt > 0
    if m.sum() == 0:
        print("no positive donations, skip.")
        return

    d = d[m]
    lt = lt[m]
    amt = amt[m]

    S_d = topic_pred["donor"][d]
    S_lt = topic_pred["legislator_term"][lt]

    sim = (S_d * S_lt).mean(dim=-1)
    log_amt = torch.log1p(amt)

    corr = _safe_corr(sim, log_amt)

    print("edges used:", int(m.sum().item()))
    print("Pearson(sim(donor, lt), log(amount)):", corr)


@torch.no_grad()
def _lobby_alignment(data, topic_pred):
    print("\n=== LOBBY ALIGNMENT CHECK ===")
    if "lobby_firm" not in topic_pred:
        print("missing lobby_firm stance, skip.")
        return

    if (
        "lobby_firm",
        "lobbied",
        "legislator_term",
    ) in data.edge_types and "legislator_term" in topic_pred:
        rel = data[("lobby_firm", "lobbied", "legislator_term")]
        if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
            lf = rel.edge_index[0]
            lt = rel.edge_index[1]
            amt = rel.edge_attr[:, 0].float()
            m = amt > 0
            if m.sum() > 0:
                lf = lf[m]
                lt = lt[m]
                amt = amt[m]
                S_lf = topic_pred["lobby_firm"][lf]
                S_lt = topic_pred["legislator_term"][lt]
                sim = (S_lf * S_lt).mean(dim=-1)
                log_amt = torch.log1p(amt)
                corr = _safe_corr(sim, log_amt)
                print(" LF-LT Pearson(sim, log(amount)):", corr)

    if (
        "lobby_firm",
        "lobbied",
        "committee",
    ) in data.edge_types and "committee" in topic_pred:
        rel = data[("lobby_firm", "lobbied", "committee")]
        if rel.edge_attr is not None and rel.edge_attr.size(0) > 0:
            lf = rel.edge_index[0]
            cm = rel.edge_index[1]
            amt = rel.edge_attr[:, 0].float()
            m = amt > 0
            if m.sum() > 0:
                lf = lf[m]
                cm = cm[m]
                amt = amt[m]
                S_lf = topic_pred["lobby_firm"][lf]
                S_cm = topic_pred["committee"][cm]
                sim = (S_lf * S_cm).mean(dim=-1)
                log_amt = torch.log1p(amt)
                corr = _safe_corr(sim, log_amt)
                print(" LF-CM Pearson(sim, log(amount)):", corr)


def quick_diagnostics(data, topic_pred, inf_pred, topic_infl, cfg):
    _inspect_basic(topic_pred, inf_pred, topic_infl, cfg.actor_types)
    _check_non_collapse(topic_pred, inf_pred, topic_infl, cfg.actor_types)
    _vote_alignment(data, topic_pred, inf_pred, cfg.num_topics)
    _donation_alignment(data, topic_pred)
    _lobby_alignment(data, topic_pred)


# ---------- Entry ----------


def run_training(data_path, pretrain_path=None):
    cfg = CFG()
    seed_all(42)

    data = torch.load(data_path, weights_only=False)
    data = ToUndirected()(data)
    data = RemoveIsolatedNodes()(data)
    attach_bill_version_labels(data, cfg.num_topics)

    model = LegModel(data, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if pretrain_path is not None:
        state_dict = torch.load(pretrain_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    loader = make_loader(data, cfg)

    for epoch in tqdm(range(1, cfg.epochs + 1)):
        loss = train_epoch(model, loader, optimizer, cfg)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f}")

    topic_pred, inf_pred, topic_infl = infer_all(model, data, cfg)

    torch.save(model.state_dict(), "leg_model_final.pth")
    torch.save(topic_pred, "topic_pred_final.pt")
    torch.save(inf_pred, "influence_pred_final.pt")
    torch.save(topic_infl, "topic_influence_final.pt")

    return data, topic_pred, inf_pred, topic_infl


def train_eval(pretrained=False):
    if pretrained:
        data, topic_pred, inf_pred, topic_infl = run_training(
            "data5.pt", pretrain_path="leg_model_final.pth"
        )
    else:
        data, topic_pred, inf_pred, topic_infl = run_training("data5.pt")
    quick_diagnostics(data, topic_pred, inf_pred, topic_infl, CFG())


if __name__ == "__main__":
    train_eval()
