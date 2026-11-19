"""D4PG with a quantile-regression critic (D4PG-QR), PyTorch.

Structure from the swing D4PG (DRL-Swing-Options/src/agent.py: local/target actor and critic,
Adam, n-step circular replay buffer, act()/step()/learn()). Learner settings follow the original
acme agent of this repository (agent/agent.py, agent/learning.py, run.py):

  * critic: 99 quantiles, quantile-Huber loss (k = 1)            ('qr-huber')
  * actor:  deterministic policy gradient on the risk objective (meanstd / var / cvar) of the
            online critic, evaluated at the NEXT observation of the sampled transition (as acme
            D4PG does), with dQ/da clipped to [-1, 1]
  * gradients clipped to global norm 40, Adam(lr = 1e-4) for both networks
  * target networks: hard copy every 100 learner steps (acme target_update_period)
  * exploration: Gaussian noise sigma = 0.3 (clipped to [-1, 1]) on the tanh action in [-1, 1]
  * replay: n-step (n = 5), discount 1, uniform sampling, learning starts after 1,000 transitions,
            one learner step per batch_size / samples_per_insert (= 32) environment steps
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .distributional import RiskObjective, quantile_huber_loss
from .networks import Actor, ObsNormalizer, QuantileCritic
from .replay_buffer import CircularReplayBuffer


class D4PGQR:
    def __init__(self, obs_dim, obj_func="var", threshold=0.95, std_coef=1.645, seed=0,
                 actor_sizes=(256, 256, 256), critic_sizes=(512, 512, 256), lr=1e-4,
                 batch_size=256, n_step=5, discount=1.0, sigma=0.3, target_update_period=100,
                 min_replay_size=1000, samples_per_insert=32.0, max_replay_size=1_000_000,
                 clip_norm=40.0, dqda_clip=1.0, obs_center=None, obs_scale=None, obs_clip=10.0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg = dict(obs_dim=obs_dim, obj_func=obj_func, threshold=threshold, std_coef=std_coef, seed=seed,
                        actor_sizes=list(actor_sizes), critic_sizes=list(critic_sizes), lr=lr,
                        batch_size=batch_size, n_step=n_step, discount=discount, sigma=sigma,
                        target_update_period=target_update_period, min_replay_size=min_replay_size,
                        samples_per_insert=samples_per_insert, max_replay_size=max_replay_size,
                        clip_norm=clip_norm, dqda_clip=dqda_clip,
                        obs_center=None if obs_center is None else [float(x) for x in obs_center],
                        obs_scale=None if obs_scale is None else [float(x) for x in obs_scale],
                        obs_clip=obs_clip)
        self.actor = Actor(obs_dim, actor_sizes, ObsNormalizer(obs_dim, obs_center, obs_scale, obs_clip))
        self.critic = QuantileCritic(obs_dim, critic_sizes, norm=ObsNormalizer(obs_dim, obs_center, obs_scale, obs_clip))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in [*self.actor_target.parameters(), *self.critic_target.parameters()]:
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-8)
        self.objective = RiskObjective(obj_func, self.critic.quantiles, self.critic.probs, threshold, std_coef)
        self.memory = CircularReplayBuffer(max_replay_size, batch_size, n_step, 1, torch.device("cpu"),
                                           seed, discount)
        self.discount_n = discount ** n_step
        self.obs_per_step = batch_size / samples_per_insert
        self.n_observed = 0
        self.learner_steps = 0
        self.last_losses = (np.nan, np.nan)
        self.train_actor = True  # False: critic-only learner steps (warm-up)

    # ------------------------------------------------------------------ acting
    @torch.inference_mode()
    def policy(self, obs, noise=False):
        """obs (B, d) numpy -> canonical action u in [-1, 1], shape (B, 1)."""
        u = self.actor(torch.as_tensor(obs, dtype=torch.float32))
        if noise:
            eps = torch.randn_like(u) * self.cfg["sigma"]
            u = (u + eps.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)  # acme ClippedGaussian + ClipToSpec
        return u.numpy()

    @staticmethod
    def to_env_action(u):
        return (np.asarray(u).reshape(-1) + 1.0) / 2.0  # acme CanonicalSpecWrapper: [-1,1] -> [0,1]

    # ------------------------------------------------------------------ learning
    def observe(self, obs, u, reward, next_obs, done):
        self.memory.add(np.asarray(obs, np.float32), np.asarray(u, np.float32).reshape(1), float(reward),
                        np.asarray(next_obs, np.float32), bool(done))
        self.n_observed += 1
        k = self.n_observed - max(self.cfg["batch_size"], self.cfg["min_replay_size"])
        if k < 0 or len(self.memory) < self.cfg["batch_size"]:
            return
        if self.obs_per_step > 1:
            if k % int(round(self.obs_per_step)) == 0:
                self.learn()
        else:
            for _ in range(int(round(1 / self.obs_per_step))):
                self.learn()

    def _hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def learn(self):
        if self.learner_steps % self.cfg["target_update_period"] == 0:
            self._hard_update()
        self.learner_steps += 1
        o_tm1, a_tm1, r_t, o_t, done, _, _ = self.memory.sample()
        d_t = self.discount_n * (1.0 - done.float())

        # critic
        with torch.no_grad():
            z_next = self.critic_target(o_t, self.actor_target(o_t))
        z = self.critic(o_tm1, a_tm1)
        critic_loss = quantile_huber_loss(z, self.critic.quantiles, r_t, d_t, z_next).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["clip_norm"])
        self.critic_opt.step()

        if not self.train_actor:
            self.last_losses = (float(critic_loss.detach()), np.nan)
            return
        # actor (trfl/acme dpg loss with dqda clipping)
        a = self.actor(o_t)
        a_det = a.detach().requires_grad_(True)
        q = self.objective(self.critic(o_t, a_det))
        dqda = torch.autograd.grad(q.sum(), a_det)[0]
        c = self.cfg["dqda_clip"]
        dqda = dqda.clamp(-c, c)
        dqda = dqda * torch.clamp(c / dqda.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        target = (a + dqda).detach()
        actor_loss = (0.5 * (target - a).pow(2).sum(-1)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["clip_norm"])
        self.actor_opt.step()
        self.critic.zero_grad(set_to_none=True)  # dqda pass left critic grads; not applied
        self.last_losses = (float(critic_loss.detach()), float(actor_loss.detach()))

    # ------------------------------------------------------------------ checkpoints
    def actor_state(self):
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def save(self, path, actor_state=None):
        torch.save(dict(cfg=self.cfg, actor=actor_state or self.actor.state_dict(),
                        critic=self.critic.state_dict()), path)

    @classmethod
    def load_actor(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        actor = Actor(cfg["obs_dim"], cfg["actor_sizes"],
                      ObsNormalizer(cfg["obs_dim"], cfg.get("obs_center"), cfg.get("obs_scale"), cfg.get("obs_clip", 10.0)))
        actor.load_state_dict(ck["actor"])
        return actor, cfg


def export_actor_npz(actor: Actor, path, meta: dict):
    """Write the policy as plain arrays (see tools/evaluate_policy_numpy.py for the forward pass)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in actor.state_dict().items()}
    arrays = {"obs_center": sd["norm_in.center"], "obs_scale": sd["norm_in.scale"],
              "obs_clip": np.array(actor.norm_in.clip),
              "W0": sd["body.first.weight"].T, "b0": sd["body.first.bias"],
              "ln_gamma": sd["body.norm.weight"], "ln_beta": sd["body.norm.bias"],
              "ln_eps": np.array(actor.body.norm.eps)}
    i = 1
    while f"body.rest.{i - 1}.weight" in sd:
        arrays[f"W{i}"] = sd[f"body.rest.{i - 1}.weight"].T
        arrays[f"b{i}"] = sd[f"body.rest.{i - 1}.bias"]
        i += 1
    arrays["W_out"] = sd["head.weight"].T
    arrays["b_out"] = sd["head.bias"]
    arrays["n_hidden"] = np.array(i)
    meta = dict(meta)
    meta.setdefault("forward", "x=(obs-obs_center)/obs_scale, clipped to +-obs_clip if obs_clip>0; "
                               "h=tanh(LayerNorm(x@W0+b0)); h=elu(h@Wi+bi) for i=1..n_hidden-1; "
                               "u=tanh(h@W_out+b_out); hedge fraction a=(u+1)/2")
    arrays["meta_json"] = np.array(json.dumps(meta, default=str))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
