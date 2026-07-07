import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================

OBS_DIM      = 13
HIDDEN_DIM   = 8
N_AGENTS     = 4
ALPHA_LEAKY  = 0.2

AGENT_ORDER = ["agent_nord", "agent_sud", "agent_est", "agent_ouest"]

# ============================================================
# COUCHE GAT (Graph Attention Layer)
# ============================================================

class GATLayer(nn.Module):
    """
    Une couche d'attention sur graphe complet (4 noeuds, tous connectes).

    Etapes :
        1. h_i' = W . h_i                      (projection)
        2. e_ij = LeakyReLU(a^T [h_i' || h_j']) (score brut)
        3. alpha_ij = softmax_j(e_ij)           (normalisation)
        4. h_i_new = sigma(Sum_j alpha_ij . h_j') (agregation ponderee)
    """

    def __init__(self, in_dim=OBS_DIM, out_dim=HIDDEN_DIM, alpha=ALPHA_LEAKY):
        super(GATLayer, self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

    def forward(self, h):
        """
        h : tensor (batch, n_agents, in_dim)
        Retourne : h_new (batch, n_agents, out_dim), alpha (batch, n_agents, n_agents)
        """
        batch_size, n_agents, _ = h.shape

        h_prime = self.W(h)  # (batch, n_agents, out_dim)

        h_i = h_prime.unsqueeze(2).expand(batch_size, n_agents, n_agents, self.out_dim)
        h_j = h_prime.unsqueeze(1).expand(batch_size, n_agents, n_agents, self.out_dim)
        concat = torch.cat([h_i, h_j], dim=-1)

        e = self.leaky_relu(self.a(concat)).squeeze(-1)  # (batch, n_agents, n_agents)
        alpha = F.softmax(e, dim=-1)

        h_new = torch.einsum("bij,bjd->bid", alpha, h_prime)
        h_new = torch.sigmoid(h_new)

        return h_new, alpha


# ============================================================
# RESEAU GAT COMPLET (multi-tetes)
# ============================================================

class GAT(nn.Module):
    """
    GAT multi-head. Retourne un etat enrichi (meme dimension obs_dim,
    grace a une connexion residuelle) pour chaque agent.
    """

    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM,
                 n_agents=N_AGENTS, n_heads=4, alpha=ALPHA_LEAKY):
        super(GAT, self).__init__()
        self.n_agents = n_agents
        self.n_heads  = n_heads

        self.heads = nn.ModuleList([
            GATLayer(in_dim=obs_dim, out_dim=hidden_dim, alpha=alpha)
            for _ in range(n_heads)
        ])

        self.output_proj = nn.Linear(n_heads * hidden_dim, obs_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.414)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, h):
        """
        h : tensor (batch, n_agents, obs_dim) -- REQUIERT grad si on entraine
        Retourne : h_enriched (batch, n_agents, obs_dim), attentions (liste)
        """
        head_outputs = []
        attentions   = []

        for head in self.heads:
            h_new, alpha = head(h)
            head_outputs.append(h_new)
            attentions.append(alpha)

        concat = torch.cat(head_outputs, dim=-1)
        h_enriched = self.output_proj(concat)

        # Connexion residuelle : garde une trace de l'etat original
        h_enriched = h_enriched + h

        return h_enriched, attentions


# ============================================================
# WRAPPER POUR L'UTILISATION AVEC LES DICTS D'AGENTS
# ============================================================

class GATModule:
    """
    Encapsule le reseau GAT. IMPORTANT : ce module n'a plus son propre
    optimizer ni son propre device choisi seul -- il est concu pour etre
    integre dans un optimizer partage avec les agents PPO (voir multi_agent.py).
    """

    def __init__(self, agent_order=AGENT_ORDER, obs_dim=OBS_DIM,
                 hidden_dim=HIDDEN_DIM, n_heads=4, alpha=ALPHA_LEAKY,
                 device=None):

        self.agent_order = agent_order
        self.n_agents    = len(agent_order)

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = GAT(
            obs_dim    = obs_dim,
            hidden_dim = hidden_dim,
            n_agents   = self.n_agents,
            n_heads    = n_heads,
            alpha      = alpha
        ).to(self.device)

    def stack_obs(self, obs_dict):
        """
        Convertit un dict {agent_id: np.array(13,)} en tensor (1, n_agents, 13)
        sur le bon device.
        """
        stacked = torch.FloatTensor(
            np.array([obs_dict[agent_id] for agent_id in self.agent_order])
        ).unsqueeze(0).to(self.device)
        return stacked

    def enrich(self, obs_dict, training=False):
        """
        Enrichit les etats des 4 agents.

        training=False (par defaut) : utilise pour la COLLECTE de donnees
            pendant le jeu -> pas de gradient necessaire, plus rapide.
        training=True : utilise PENDANT l'update PPO -> garde le graphe
            de calcul pour que le gradient remonte jusqu'au GAT.

        Retourne :
        enriched_dict : dict {agent_id: np.array(13,)}  (toujours numpy, pour le jeu)
        attentions    : liste de matrices d'attention
        """
        stacked = self.stack_obs(obs_dict)

        if training:
            h_enriched, attentions = self.network(stacked)
        else:
            with torch.no_grad():
                h_enriched, attentions = self.network(stacked)

        h_enriched_np = h_enriched.detach().squeeze(0).cpu().numpy()

        enriched_dict = {
            agent_id: h_enriched_np[i]
            for i, agent_id in enumerate(self.agent_order)
        }

        return enriched_dict, attentions

    def enrich_batch(self, obs_batch_dict):
        """
        Version BATCH pour l'update PPO : enrichit un batch entier d'etats
        en gardant le graphe de calcul (utilise dans PPOAgent.update via
        MultiAgentPPO).

        obs_batch_dict : dict {agent_id: tensor (N, 13)} -- deja sur device, requires_grad pas necessaire en entree

        Retourne :
        enriched_batch : dict {agent_id: tensor (N, 13)} -- CONNECTE au graphe de calcul du GAT
        """
        # Empiler : (N, n_agents, 13)
        stacked = torch.stack(
            [obs_batch_dict[agent_id] for agent_id in self.agent_order],
            dim=1
        )  # (N, n_agents, 13)

        h_enriched, attentions = self.network(stacked)  # garde le grad

        enriched_batch = {
            agent_id: h_enriched[:, i, :]
            for i, agent_id in enumerate(self.agent_order)
        }

        return enriched_batch, attentions