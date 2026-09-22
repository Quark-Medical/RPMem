"""CPU-only gate-training example using small, randomly initialized modules.

Run after installing the package: python examples/core_smoke.py
No pretrained weights, dataset, GPU, or network access is needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from rpmem import CMPGate, RPMemConfig, HeadConfig, PerceiverConfig
from rpmem.head.hypernet_head import HyperLoRAHead
from rpmem.model import RPMemModel
from rpmem.training.hypernet_model import HypernetModel
from rpmem.training.losses import mcq_ce_loss
from rpmem.training.trainer import CMPTrainer


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(8, 8)})

    def forward(self, x):
        return x + torch.tanh(self.mlp["down_proj"](x))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def make_model():
    config = RPMemConfig(
        base_model_name="tiny-random-reader", n_layers=2, layer_indices=[0, 1],
        perceiver=PerceiverConfig(
            input_size=8, hidden_size=8, n_latent_queries=2,
            num_attention_heads=2, num_key_value_heads=2, encoder_num_blocks=1,
        ),
        head=HeadConfig(
            d_latent=8, n_layers=2, r=2, num_pre_head_layers=1,
            in_features={"down_proj": 8}, out_features={"down_proj": 8},
        ),
    )
    head = HyperLoRAHead(config.head)
    # Represent a nonzero trained adapter scale so the gate receives a signal.
    with torch.no_grad():
        head.scaler_B["down_proj"].fill_(0.1)
    return RPMemModel(
        config, TinyBackbone(), nn.Identity(),
        HypernetModel._build_perceiver(config), head, CMPGate(d_latent=8),
    )


def main():
    torch.manual_seed(7)
    model = make_model()
    model.eval()
    with torch.no_grad():
        features = torch.randn(3, 2, 6, 8)
        mask = torch.ones(3, 6, dtype=torch.long)
        latents, _ = model.perceiver(features, mask, None)
        sessions = list(latents.split(1, dim=0))
    model.setup_training()
    trainer = CMPTrainer(
        model.base_model, model.head, model.gate, [0, 1], ["down_proj"],
        device="cpu", lora_alpha=model.config.lora.lora_alpha,
    )
    frozen = {n: p.detach().clone() for n, p in model.named_parameters()
              if not n.startswith("gate.")}
    before = model.gate.gate.weight.detach().clone()
    ids = torch.tensor([[1, 2, 3]])
    losses = [trainer.train_step(sessions, ids, lambda x: mcq_ce_loss(x, 4))
              for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(before, model.gate.gate.weight)
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value)
    print("Gate training passed; compiler and backbone remained frozen.")
    print("Memory shape:", tuple(model.merge_sessions(sessions).shape))
    print("Losses:", [round(x, 6) for x in losses])


if __name__ == "__main__":
    main()
