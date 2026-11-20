"""Risk-sensitive gate on top of a rule: quantile critic over two actions (trade the rule amount / no trade).

Same critic network, quantile-Huber loss, n-step replay and risk objective (VaR95 of the return
distribution) as D4PGQR; the actor is replaced by a greedy choice between the two actions:
    g(s) = argmax_{g in {0,1}} objective(Z(s, g))        (objective = 5% quantile of the return for VaR95)
Target: Z_target(s', g*(s')), g* greedy on the online critic (double-DQN style).
Warm start: during the first episodes the agent plays the rule's own gate (rule (b)'s band) with
epsilon exploration, so the critic learns on rule data before the gate is used.
Conservative switch: deviate from the rule's gate only when the objective gain is > margin.
"""
import copy

import numpy as np
import torch

from .distributional import RiskObjective, quantile_huber_loss
from .networks import ObsNormalizer, QuantileCritic
from .replay_buffer import CircularReplayBuffer


class GateQR:
    def __init__(self, obs_dim, obj_func="var", threshold=0.95, std_coef=1.645, seed=0,
                 critic_sizes=(512, 512, 256), lr=1e-4, batch_size=256, n_step=5, discount=1.0,
                 target_update_period=100, min_replay_size=1000, samples_per_insert=32.0,
                 max_replay_size=1_000_000, clip_norm=40.0, obs_center=None, obs_scale=None, obs_clip=10.0,
                 epsilon=0.1, margin=0.0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg = dict(obs_dim=obs_dim, obj_func=obj_func, threshold=threshold, seed=seed,
                        critic_sizes=list(critic_sizes), lr=lr, batch_size=batch_size, n_step=n_step,
                        target_update_period=target_update_period, min_replay_size=min_replay_size,
                        samples_per_insert=samples_per_insert, clip_norm=clip_norm, epsilon=epsilon, margin=margin,
                        obs_center=None if obs_center is None else [float(x) for x in obs_center],
                        obs_scale=None if obs_scale is None else [float(x) for x in obs_scale], obs_clip=obs_clip)
        self.critic = QuantileCritic(obs_dim, critic_sizes, norm=ObsNormalizer(obs_dim, obs_center, obs_scale, obs_clip))
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-8)
        self.objective = RiskObjective(obj_func, self.critic.quantiles, self.critic.probs, threshold, std_coef)
        self.memory = CircularReplayBuffer(max_replay_size, batch_size, n_step, 1, torch.device("cpu"), seed, discount)
        self.discount_n = discount ** n_step
        self.obs_per_step = batch_size / samples_per_insert
        self.n_observed = 0
        self.learner_steps = 0
        self.last_losses = (np.nan, np.nan)
        self.use_critic = False  # False: play the rule's gate (warm-up)

    def _scores(self, critic, obs):
        B = obs.shape[0]
        s0 = self.objective(critic(obs, torch.zeros(B, 1)))
        s1 = self.objective(critic(obs, torch.ones(B, 1)))
        return s0, s1

    @torch.inference_mode()
    def act(self, obs, rule_gate, explore=False):
        """obs (B, d), rule_gate (B,) in {0,1} -> gate (B,) in {0,1}."""
        g = np.asarray(rule_gate, dtype=float).copy()
        if self.use_critic:
            s0, s1 = self._scores(self.critic, torch.as_tensor(obs, dtype=torch.float32))
            s0, s1 = s0.numpy(), s1.numpy()
            m = self.cfg["margin"]
            g = np.where(g == 1, np.where(s0 > s1 + m, 0.0, 1.0), np.where(s1 > s0 + m, 1.0, 0.0))
        if explore:
            flip = self.rng.random(len(g)) < self.cfg["epsilon"]
            g = np.where(flip, 1.0 - g, g)
        return g

    def observe(self, obs, g, reward, next_obs, done):
        self.memory.add(np.asarray(obs, np.float32), np.asarray([g], np.float32), float(reward),
                        np.asarray(next_obs, np.float32), bool(done))
        self.n_observed += 1
        k = self.n_observed - max(self.cfg["batch_size"], self.cfg["min_replay_size"])
        if k < 0 or len(self.memory) < self.cfg["batch_size"]:
            return
        if k % max(1, int(round(self.obs_per_step))) == 0:
            self.learn()

    def learn(self):
        if self.learner_steps % self.cfg["target_update_period"] == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
        self.learner_steps += 1
        o_tm1, a_tm1, r_t, o_t, done, _, _ = self.memory.sample()
        d_t = self.discount_n * (1.0 - done.float())
        with torch.no_grad():
            # greedy next action on the online critic (no rule prior in the target: Q-learning of the gate)
            s0, s1 = self._scores(self.critic, o_t)
            g_next = (s1 > s0).float().unsqueeze(-1)
            z_next = self.critic_target(o_t, g_next)
        z = self.critic(o_tm1, a_tm1)
        loss = quantile_huber_loss(z, self.critic.quantiles, r_t, d_t, z_next).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["clip_norm"])
        self.opt.step()
        self.last_losses = (float(loss.detach()), np.nan)

    def state(self):
        return {k: v.detach().clone() for k, v in self.critic.state_dict().items()}
