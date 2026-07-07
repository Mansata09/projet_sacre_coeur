import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ============================================================
# HYPERPARAMETRES PPO
# ============================================================

GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
COEF_VF      = 0.5
COEF_ENTROPY = 0.01
N_EPOCHS     = 4
BATCH_SIZE   = 64
BUFFER_SIZE  = 2048

OBS_DIM      = 13
ACT_DIM      = 4
HIDDEN_DIM   = 64

# ============================================================
# RESEAU ACTOR-CRITIC
# ============================================================

class ActorCritic(nn.Module):
    """
    Reseau partage Actor-Critic.

    Shared : Linear(13->64) -> ReLU -> Linear(64->64) -> ReLU
    Actor  : Linear(64->4)  -> Softmax -> pi(a|s)
    Critic : Linear(64->1)             -> V(s)
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor  = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h      = self.shared(obs)
        logits = self.actor(h)
        value  = self.critic(h)
        return logits, value

    def get_action(self, obs):
        """
        obs : tensor (1, 13) -- utilise pendant le JEU (pas de grad necessaire en amont)
        """
        logits, value = self.forward(obs)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        entropy       = dist.entropy()
        return action.item(), log_prob, value.squeeze(-1), entropy

    def evaluate(self, obs, actions):
        """
        obs : tensor (N, 13) -- DOIT garder le graphe de calcul pendant l'update
        (obs vient du GAT, donc le gradient remonte naturellement)
        """
        logits, values = self.forward(obs)
        dist           = Categorical(logits=logits)
        log_probs      = dist.log_prob(actions)
        entropy        = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ============================================================
# BUFFER D'EXPERIENCES
# ============================================================

class PPOBuffer:
    """
    Stocke les experiences (s_brut, a, r, log_prob, value, done).

    IMPORTANT : on stocke l'etat BRUT (sortie de state.py), PAS l'etat
    enrichi par le GAT. L'enrichissement est refait au moment de l'update
    pour que le gradient remonte jusqu'au GAT (voir multi_agent.py).
    """

    def __init__(self, obs_dim=OBS_DIM, buffer_size=BUFFER_SIZE,
                    gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        self.obs_dim     = obs_dim
        self.buffer_size = buffer_size
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.reset()

    def reset(self):
        self.obs       = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions   = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards   = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.values    = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones     = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr       = 0
        self.full      = False

    def store(self, obs_brut, action, reward, log_prob, value, done):
        """obs_brut : etat AVANT enrichissement GAT (13 features de state.py)"""
        self.obs      [self.ptr] = obs_brut
        self.actions  [self.ptr] = action
        self.rewards  [self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values   [self.ptr] = value
        self.dones    [self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value=0.0):
        """
        delta_t = r_t + gamma V(s_t+1) - V(s_t)
        A_t = Sum (gamma*lambda)^k delta_t+k
        R_t = A_t + V(s_t)
        """
        advantages = np.zeros(self.ptr, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done  = 0.0
            else:
                next_value = self.values[t + 1]
                next_done  = self.dones [t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + self.values[:self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def is_ready(self):
        return self.full


# ============================================================
# AGENT PPO (réseau seul -- l'optimizer est gere par MultiAgentPPO)
# ============================================================

class PPOAgent:
    """
    Un agent PPO pour UNE branche.

    IMPORTANT (architecture GAT+PPO conjointe) : cet agent ne possede plus
    son propre optimizer. Le réseau ActorCritic est cree ici, mais c'est
    MultiAgentPPO qui collecte les parametres de tous les agents + du GAT
    dans UN SEUL optimizer partage, pour que le gradient de la loss PPO
    remonte correctement jusqu'aux poids du GAT.
    """

    def __init__(self, agent_id, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                    hidden_dim=HIDDEN_DIM, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                    clip_eps=CLIP_EPS, coef_vf=COEF_VF, coef_entropy=COEF_ENTROPY,
                    buffer_size=BUFFER_SIZE, device=None):

        self.agent_id     = agent_id
        self.clip_eps     = clip_eps
        self.coef_vf      = coef_vf
        self.coef_entropy = coef_entropy

        # PPO tourne toujours sur CPU (reseau petit, pas besoin de GPU)
        self.device = device if device is not None else torch.device("cpu")

        self.network = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.buffer  = PPOBuffer(obs_dim, buffer_size, gamma, gae_lambda)

    def select_action(self, obs_enriched):
        """
        obs_enriched : np.array (13,) -- etat APRES passage par le GAT
        (utilise seulement pour CHOISIR l'action pendant le jeu, no_grad)
        """
        obs_t = torch.FloatTensor(obs_enriched).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action(obs_t)
        return action, log_prob.item(), value.item()

    def store(self, obs_brut, action, reward, log_prob, value, done):
        self.buffer.store(obs_brut, action, reward, log_prob, value, done)

    def prepare_epoch(self, last_value=0.0):
        """
        A appeler UNE FOIS par epoch (pas par mini-batch), car GAE doit etre
        calcule sur la trajectoire complete, pas sur un sous-ensemble.

        Stocke advantages/returns/actions/old_log_probs en interne pour que
        compute_loss_minibatch puisse les indexer ensuite.
        """
        advantages, returns = self.buffer.compute_gae(last_value)

        self._actions_t  = torch.LongTensor(self.buffer.actions[:self.buffer.ptr]).to(self.device)
        self._old_lp_t   = torch.FloatTensor(self.buffer.log_probs[:self.buffer.ptr]).to(self.device)
        self._adv_t      = torch.FloatTensor(advantages).to(self.device)
        self._returns_t  = torch.FloatTensor(returns).to(self.device)
        # Valeurs predites par le Critic AU MOMENT DE LA COLLECTE (avant tout
        # update de cette epoch) -- necessaire pour le value clipping.
        self._old_values_t = torch.FloatTensor(self.buffer.values[:self.buffer.ptr]).to(self.device)

    def compute_loss_minibatch(self, obs_enriched_batch, idx, last_value=0.0):
        """
        Calcule la loss PPO sur un MINI-BATCH (indices idx), en gardant le
        graphe de calcul (obs_enriched_batch vient du GAT avec grad actif).

        obs_enriched_batch : tensor (len(idx), 13), connecte au graphe du GAT
        idx                : indices numpy du mini-batch dans le buffer complet

        Retourne :
        loss      : scalaire tensor, AVEC grad
        approx_kl : float, SANS grad -- KL-divergence approximee entre
                    l'ancienne et la nouvelle politique sur ce mini-batch,
                    utilisee pour l'early stopping (voir multi_agent.py).
                    Estimateur non biaise et a faible variance
                    (Schulman, http://joschu.net/blog/kl-approx.html) :
                    approx_kl = E[(ratio - 1) - log_ratio]
        """
        idx_t = torch.LongTensor(idx).to(self.device)

        actions_batch    = self._actions_t   [idx_t]
        old_lp_batch     = self._old_lp_t    [idx_t]
        adv_batch        = self._adv_t       [idx_t]
        returns_batch    = self._returns_t   [idx_t]
        old_values_batch = self._old_values_t[idx_t]

        new_log_probs, values, entropy = self.network.evaluate(obs_enriched_batch, actions_batch)

        log_ratio = new_log_probs - old_lp_batch
        ratio     = torch.exp(log_ratio)

        # ---- KL approximee (pour early stopping dans MultiAgentPPO.update) ----
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()

        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
        loss_actor = -torch.min(surr1, surr2).mean()

        # ---- Value loss, normalisee ----
        # Avec gamma proche de 1 et des rewards constamment negatifs (cas
        # frequent ici : penalites de trafic), les returns GAE peuvent
        # atteindre des centaines en valeur absolue meme sur un buffer
        # de taille modeste. Le MSE brut (erreur au carre) explose alors
        # mecaniquement. On normalise returns ET values (meme echelle,
        # memes statistiques calculees sur returns_batch) avant le MSE,
        # ce qui ramene la loss_critic a un ordre de grandeur stable
        # (proche de 1), sans changer la direction du gradient.
        ret_mean = returns_batch.mean()
        ret_std  = returns_batch.std() + 1e-8

        values_norm    = (values - ret_mean) / ret_std
        returns_norm   = (returns_batch - ret_mean) / ret_std
        old_values_norm = (old_values_batch - ret_mean) / ret_std

        values_clipped_norm = old_values_norm + torch.clamp(
            values_norm - old_values_norm, -self.clip_eps, self.clip_eps
        )
        loss_unclipped = (values_norm - returns_norm) ** 2
        loss_clipped   = (values_clipped_norm - returns_norm) ** 2
        loss_critic    = self.coef_vf * torch.max(loss_unclipped, loss_clipped).mean()

        loss_entropy = -self.coef_entropy * entropy.mean()

        loss = loss_actor + loss_critic + loss_entropy
        return loss, approx_kl

    def reset_buffer(self):
        self.buffer.reset()