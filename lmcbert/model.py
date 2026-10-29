"""
LMCBert (Liu et al., IEEE TCYB 2025) — BERT classifier + MoCo contrastive branch.

  q  = norm(BERT(summary).pooler_output)          (query encoder, trained)
  k  = norm(BERT_m(abstract).pooler_output)       (momentum encoder, EMA of BERT)
  logits = [q·k, q·queue] / tau ; label 0        (InfoNCE, Alg. 1)
  L = lambda * L_contrastive + (1 - lambda) * BCE(Linear(BERT(summary)), y)

Variants (for the ablations in Table III):
  contrastive="moco"     full model (queue of size K, momentum m)
  contrastive="inbatch"  "w/o queue": negatives are the other keys in the batch
  contrastive="none"     "w/o contrastive": plain fine-tuned BERT
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LMCBert(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased", queue_size=4096, momentum=0.99, tau=0.07,
                 contrastive="moco", config=None, extra_dim=0, aux_head=False):
        """
        extra_dim : number of side features (structural / novelty) fused into the
                    classifier — enhancement modules from the paper's conclusion.
        aux_head  : add a regression head predicting the mean reviewer
                    recommendation score (expert-annotated data, e.g. PeerRead).
        """
        super().__init__()
        if config is not None:  # for offline smoke tests with a random tiny BERT
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        hid = self.encoder.config.hidden_size
        self.extra_dim = extra_dim
        if extra_dim:
            self.feat_proj = nn.Sequential(nn.Linear(extra_dim, 64), nn.GELU(), nn.Dropout(0.1))
            self.classifier = nn.Linear(hid + 64, 1)
        else:
            self.classifier = nn.Linear(hid, 1)
        self.aux_head = nn.Linear(hid, 1) if aux_head else None
        self.contrastive = contrastive
        self.tau = tau
        self.momentum = momentum
        if contrastive == "moco":
            self.encoder_m = copy.deepcopy(self.encoder)
            for p in self.encoder_m.parameters():
                p.requires_grad = False
            self.register_buffer("queue", F.normalize(torch.randn(hid, queue_size), dim=0))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        elif contrastive == "inbatch":
            self.encoder_m = copy.deepcopy(self.encoder)
            for p in self.encoder_m.parameters():
                p.requires_grad = False

    # ---------------------------------------------------------------- helpers
    def _pool(self, enc, **inputs):
        out = enc(**inputs)
        return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]

    @torch.no_grad()
    def _momentum_update(self):
        for p, pm in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            pm.data.mul_(self.momentum).add_(p.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, k):
        bs = k.shape[0]
        K = self.queue.shape[1]
        ptr = int(self.queue_ptr)
        if ptr + bs <= K:
            self.queue[:, ptr:ptr + bs] = k.T
        else:  # wrap around
            first = K - ptr
            self.queue[:, ptr:] = k[:first].T
            self.queue[:, : bs - first] = k[first:].T
        self.queue_ptr[0] = (ptr + bs) % K

    # ---------------------------------------------------------------- forward
    def forward(self, summary_inputs, abstract_inputs=None, labels=None, lam=0.5,
                extra=None, aux_target=None, aux_weight=0.1):
        s_f = self._pool(self.encoder, **summary_inputs)
        h = torch.cat([s_f, self.feat_proj(extra)], dim=1) if self.extra_dim else s_f
        logit = self.classifier(h).squeeze(-1)
        out = {"logit": logit}
        if self.aux_head is not None:
            out["aux"] = self.aux_head(s_f).squeeze(-1)
        if labels is None:
            return out
        l_a = F.binary_cross_entropy_with_logits(logit, labels.float())
        if self.aux_head is not None and aux_target is not None:
            mask = ~torch.isnan(aux_target)
            if mask.any():
                l_aux = F.mse_loss(out["aux"][mask], aux_target[mask])
                out["loss_aux"] = l_aux
                l_a = l_a + aux_weight * l_aux
        out["loss_cls"] = l_a
        if self.contrastive == "none" or abstract_inputs is None:
            out["loss"] = l_a
            return out

        q = F.normalize(s_f, dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self._pool(self.encoder_m, **abstract_inputs), dim=1)

        if self.contrastive == "moco":
            pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
            neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
            logits = torch.cat([pos, neg], dim=1) / self.tau
            targets = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            l_c = F.cross_entropy(logits, targets)
            self._enqueue(k)
        else:  # in-batch negatives (w/o queue)
            logits = q @ k.T / self.tau
            targets = torch.arange(q.shape[0], device=q.device)
            l_c = F.cross_entropy(logits, targets)
        out["loss_con"] = l_c
        out["loss"] = lam * l_c + (1.0 - lam) * l_a
        return out
