"""D4PG-QR hedging agent trained with the swing-options "v64" recipe (DRL-Swing-Options/run.py defaults).

Experiment: does the training recipe of the swing-options D4PG close the gap to the tuned banded rule?
The swing agent optimises the MEAN with a scalar MSE critic, so it cannot optimise VaR as it is.
We keep what the hedging problem needs (quantile critic, 99 levels, quantile-Huber loss, VaR95 actor
objective, robust median/IQR state scaling) and take everything else from the swing v64 defaults:

  * networks: 3 hidden layers x 64, LayerNorm + SiLU, orthogonal init       (--actor/critic_layers 3, -layer_size 64)
  * actor output: beta-sigmoid, a = sigmoid(1.5 u) in [0, 1]                (--actor_output_activation beta_sigmoid_1.5)
  * AdamW, lr_actor 3e-4 (wd 5e-5), lr_critic 6e-4 (wd 1e-4), no grad clip  (-lr_a, -lr_c, --optimizer adamw)
  * Polyak target updates, tau = 0.0032                                     (-t 0.0032)
  * 1-step TD, discount 1, batch 128, replay 1e5, learning starts at 1,000  (-nstep 1, -bs 128)
  * learn every 2 env steps, 2 gradient steps each                          (-learn_every 2, -learn_number 2)
  * plain DPG actor loss -objective(Z(s, actor(s)))                         (swing learn_)
  * critic warm-up: no actor updates for the first 512 episodes             (--critic_warmup_episodes 512)
  * pre-squash Gaussian noise, linear 1.30 -> 0.26 over training,
    scaled by (1 + 0.6 |u|); during warm-up ramped from 0.3x to 1x           (-noise_sigma0/-noise_floor, --adaptive_noise_scale,
                                                                             --warmup_noise_fraction, --noise_schedule linear)
  * evaluation with an EMA (0.999) of the actor weights                     (--weight_averaging ema)
Swing-only parts NOT used: profitability gate, HHK input layer, semi-analytical kernel target,
closed-form bias warm-start (all specific to the swing contract / HHK dynamics).
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .distributional import RiskObjective, quantile_huber_loss
from .networks import ObsNormalizer, quantile_grid
from .replay_buffer import CircularReplayBuffer


class SwingMLP(nn.Module):
    """[Linear -> LayerNorm -> SiLU] x n, orthogonal init (gain for SiLU ~ ReLU gain sqrt(2))."""

    def __init__(self, in_dim, width=64, layers=3):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            lin = nn.Linear(d, width)
            nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            mods += [lin, nn.LayerNorm(width), nn.SiLU()]
            d = width
        self.net = nn.Sequential(*mods)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class SwingActor(nn.Module):
    def __init__(self, obs_dim, width=64, layers=3, beta=1.5, norm=None):
        super().__init__()
        self.norm_in = norm if norm is not None else ObsNormalizer(obs_dim)
        self.body = SwingMLP(obs_dim, width, layers)
        self.head = nn.Linear(self.body.out_dim, 1)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)
        self.beta = float(beta)

    def preact(self, obs):
        return self.head(self.body(self.norm_in(obs)))

    def squash(self, u):
        return torch.sigmoid(self.beta * u)

    def forward(self, obs):  # hedge fraction in [0, 1]
        return self.squash(self.preact(obs))


class SwingQuantileCritic(nn.Module):
    def __init__(self, obs_dim, width=64, layers=3, norm=None):
        super().__init__()
        self.norm_in = norm if norm is not None else ObsNormalizer(obs_dim)
        q, p = quantile_grid(0.01)
        self.register_buffer("quantiles", torch.tensor(q, dtype=torch.float32))
        self.register_buffer("probs", torch.tensor(p, dtype=torch.float32))
        self.body = SwingMLP(obs_dim + 1, width, layers)
        self.head = nn.Linear(self.body.out_dim, len(q))
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, action):  # action = hedge fraction in [0, 1]
        return self.head(self.body(torch.cat([self.norm_in(obs), action], dim=-1)))


class D4PGQRSwing:
    def __init__(self, obs_dim, obj_func="var", threshold=0.95, std_coef=1.645, seed=0, total_episodes=40_000,
                 width=64, layers=3, beta=1.5, lr_a=3e-4, lr_c=6e-4, wd_a=5e-5, wd_c=1e-4, tau=0.0032,
                 batch_size=128, n_step=1, max_replay_size=100_000, min_replay_size=1000,
                 learn_every=2, learn_number=2, critic_warmup_episodes=512, noise_sigma0=1.30,
                 noise_floor=0.26, adaptive_noise_scale=0.6, warmup_noise_fraction=0.3, ema_decay=0.999,
                 obs_center=None, obs_scale=None, obs_clip=10.0):
        torch.manual_seed(seed)
        self.cfg = {k: v for k, v in locals().items() if k not in ("self", "obs_center", "obs_scale")}
        self.cfg.update(obs_center=None if obs_center is None else [float(x) for x in obs_center],
                        obs_scale=None if obs_scale is None else [float(x) for x in obs_scale])

        def nrm():
            return ObsNormalizer(obs_dim, obs_center, obs_scale, obs_clip)

        self.actor = SwingActor(obs_dim, width, layers, beta, nrm())
        self.critic = SwingQuantileCritic(obs_dim, width, layers, nrm())
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_ema = copy.deepcopy(self.actor)  # eval-only weight average
        for m in (self.actor_target, self.critic_target, self.actor_ema):
            for p in m.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr_a, weight_decay=wd_a)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr_c, weight_decay=wd_c)
        self.objective = RiskObjective(obj_func, self.critic.quantiles, self.critic.probs, threshold, std_coef)
        self.memory = CircularReplayBuffer(max_replay_size, batch_size, n_step, 1, torch.device("cpu"), seed, 1.0)
        self.episode = 0
        self.n_observed = 0
        self.learner_steps = 0
        self.last_losses = (np.nan, np.nan)

    # ------------------------------------------------------------------ schedule
    def start_episode(self, ep):
        self.episode = ep + 1  # 1-based as the swing agent's episode counter

    def noise_sigma(self):
        c = self.cfg
        e = max(1, self.episode)
        frac = min(1.0, e / max(1, c["total_episodes"]))
        s = c["noise_sigma0"] + (c["noise_floor"] - c["noise_sigma0"]) * frac
        if e <= c["critic_warmup_episodes"] and c["warmup_noise_fraction"] < 1.0:
            w = e / max(1, c["critic_warmup_episodes"])
            s *= c["warmup_noise_fraction"] + (1.0 - c["warmup_noise_fraction"]) * w
        return s

    # ------------------------------------------------------------------ acting
    @torch.inference_mode()
    def policy(self, obs, noise=False):
        """obs (B, d) -> hedge fraction in [0, 1], shape (B, 1)."""
        u = self.actor.preact(torch.as_tensor(obs, dtype=torch.float32))
        if noise:
            s = self.noise_sigma()
            u = u + torch.randn_like(u) * s * (1.0 + self.cfg["adaptive_noise_scale"] * u.abs())
        return self.actor.squash(u).numpy()

    @staticmethod
    def to_env_action(a):
        return np.asarray(a).reshape(-1)

    # ------------------------------------------------------------------ learning
    def observe(self, obs, a, reward, next_obs, done):
        self.memory.add(np.asarray(obs, np.float32), np.asarray(a, np.float32).reshape(1), float(reward),
                        np.asarray(next_obs, np.float32), bool(done))
        self.n_observed += 1
        c = self.cfg
        if len(self.memory) < c["min_replay_size"] or len(self.memory) <= c["batch_size"]:
            return
        if self.n_observed % c["learn_every"] != 0:
            return
        for _ in range(c["learn_number"]):
            self.learn()

    def _soft(self, src, dst, tau):
        with torch.no_grad():
            for d, s in zip(dst.parameters(), src.parameters()):
                d.mul_(1 - tau).add_(s, alpha=tau)

    def learn(self):
        c = self.cfg
        self.learner_steps += 1
        o, a, r, o2, done, _, _ = self.memory.sample()
        d = 1.0 - done.float()
        with torch.no_grad():
            z_next = self.critic_target(o2, self.actor_target(o2))
        z = self.critic(o, a)
        critic_loss = quantile_huber_loss(z, self.critic.quantiles, r, d, z_next).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_val = 0.0
        if self.episode > c["critic_warmup_episodes"]:
            actor_loss = -self.objective(self.critic(o, self.actor(o))).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic.zero_grad(set_to_none=True)
            actor_loss_val = float(actor_loss.detach())
            self._soft(self.actor, self.actor_target, c["tau"])
            self._soft(self.actor, self.actor_ema, 1.0 - c["ema_decay"])
        self._soft(self.critic, self.critic_target, c["tau"])
        self.last_losses = (float(critic_loss.detach()), actor_loss_val)

    # ------------------------------------------------------------------ checkpoints
    def eval_actor(self):
        return self.actor_ema

    def actor_state(self):
        return {k: v.detach().clone() for k, v in self.actor_ema.state_dict().items()}

    def save(self, path, actor_state=None):
        torch.save(dict(cfg=self.cfg, actor=actor_state or self.actor_ema.state_dict(),
                        critic=self.critic.state_dict()), path)
