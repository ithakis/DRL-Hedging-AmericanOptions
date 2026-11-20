"""D4PG-QR with the expected (kernel) critic target of torch_d4pg/heston_kernel.py.

Differences to D4PGQR (everything else, incl. the actor update, is inherited):
  * 1-step transitions; for each one the M quadrature nodes are computed ONCE when it is stored
    (reward r_j and next state s'_j at node j; they depend on the hedge holdings, known at that time)
  * critic loss = sum_j w_j * quantile-Huber(Z(s, a), r_j + Z_target(s'_j, pi_target(s'_j)))
    (exact for the mixture target because the quantile loss is linear in the target distribution)
  * the actor is updated at the realised next state (as D4PGQR, acme style)
"""
import numpy as np
import torch

from .agent import D4PGQR
from .distributional import quantile_huber_loss


class KernelReplay:
    def __init__(self, size, obs_dim, M, seed):
        self.size, self.pos, self.n = size, 0, 0
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, 1), np.float32)
        self.r = np.zeros((size, M), np.float32)
        self.o1 = np.zeros((size, M, obs_dim), np.float32)
        self.o1_real = np.zeros((size, obs_dim), np.float32)
        self.w = np.zeros((size, M), np.float32)
        self.d = np.zeros(size, np.float32)
        self.rng = np.random.RandomState(seed)

    def add(self, o, a, r, o1, o1_real, d, w):
        k = len(o)
        idx = (self.pos + np.arange(k)) % self.size
        self.o[idx], self.a[idx], self.r[idx], self.o1[idx], self.o1_real[idx], self.d[idx] = o, a, r, o1, o1_real, d
        self.w[idx] = w
        self.pos = (self.pos + k) % self.size
        self.n = min(self.n + k, self.size)

    def __len__(self):
        return self.n

    def sample(self, B):
        i = self.rng.randint(0, self.n, B)
        t = lambda x: torch.from_numpy(x[i])
        return t(self.o), t(self.a), t(self.r), t(self.o1), t(self.o1_real), t(self.d), t(self.w)


class D4PGKernel(D4PGQR):
    def __init__(self, obs_dim, n_nodes, **kw):
        kw["n_step"] = 1
        super().__init__(obs_dim, **kw)
        self.M = n_nodes
        self.kmem = KernelReplay(kw.get("max_replay_size", 1_000_000), obs_dim, self.M, kw.get("seed", 0))

    def observe_batch(self, o, u, r_nodes, o1_nodes, o1_real, done, w_nodes):
        """K transitions at once (one per parallel path)."""
        K = len(o)
        rs = self.cfg["reward_scale"]
        self.kmem.add(o, u.reshape(K, 1), r_nodes * rs, o1_nodes, o1_real, np.full(K, float(done)), w_nodes)
        for _ in range(K):
            self.n_observed += 1
            k = self.n_observed - max(self.cfg["batch_size"], self.cfg["min_replay_size"])
            if k < 0 or len(self.kmem) < self.cfg["batch_size"]:
                continue
            if self.obs_per_step > 1:
                if k % int(round(self.obs_per_step)) == 0:
                    self.learn()
            else:
                for _ in range(int(round(1 / self.obs_per_step))):
                    self.learn()

    def learn(self):
        if self.cfg["tau"] > 0:
            if self.learner_steps == 0:
                self._hard_update()
        elif self.learner_steps % self.cfg["target_update_period"] == 0:
            self._hard_update()
        self.learner_steps += 1
        o, a, r, o1, o1_real, d, w = self.kmem.sample(self.cfg["batch_size"])
        B, M, D = o1.shape
        with torch.no_grad():
            flat = o1.reshape(B * M, D)
            z_next = self.critic_target(flat, self.actor_target(flat))           # (B*M, N)
        z = self.critic(o, a)                                                   # (B, N)
        zr = z.repeat_interleave(M, dim=0)
        disc = (1.0 - d).repeat_interleave(M).unsqueeze(-1)
        loss_bm = quantile_huber_loss(zr, self.critic.quantiles, r.reshape(B * M, 1), disc, z_next).reshape(B, M)
        critic_loss = (loss_bm * w).sum(1).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["clip_norm"])
        self.critic_opt.step()
        if not self.train_actor:
            self.last_losses = (float(critic_loss.detach()), np.nan)
            self._after_step()
            return
        o_t = o1_real
        act = self.actor(o_t)
        a_det = act.detach().requires_grad_(True)
        q = self.objective(self.critic(o_t, a_det))
        dqda = torch.autograd.grad(q.sum(), a_det)[0]
        c = self.cfg["dqda_clip"]
        dqda = dqda.clamp(-c, c)
        dqda = dqda * torch.clamp(c / dqda.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        target = (act + dqda).detach()
        actor_loss = (0.5 * (target - act).pow(2).sum(-1)).mean()
        if self.cfg["residual_l2"] > 0:
            actor_loss = actor_loss + self.cfg["residual_l2"] * (0.5 * act.pow(2).sum(-1)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["clip_norm"])
        self.actor_opt.step()
        self.critic.zero_grad(set_to_none=True)
        self.last_losses = (float(critic_loss.detach()), float(actor_loss.detach()))
        self._after_step()
