import numpy as np
import os
import torch

from agents.ppo_agent import PPOAgent
from agents.gat import GATModule

# ============================================================
# CONFIGURATION
# ============================================================

AGENT_IDS = ["agent_nord", "agent_sud", "agent_est", "agent_ouest"]
MODELS_DIR = os.path.join("..", "models")

LR           = 3e-4
N_EPOCHS     = 4
BATCH_SIZE   = 64
TARGET_KL    = 0.02

# ============================================================
# MULTI-AGENT PPO + GAT (entrainement conjoint)
# ============================================================

class MultiAgentPPO:
    """
    Gere :
    - 1 GAT (partage entre les 4 agents)
    - 4 PPOAgent (un par branche)
    - 1 SEUL optimizer Adam sur TOUS les parametres (GAT + 4 ActorCritic)

    C'est cet optimizer unique qui permet au gradient de la loss PPO de
    remonter jusqu'aux poids du GAT pendant l'update.
    """

    def __init__(self, agent_ids=AGENT_IDS, obs_dim=13, act_dim=4,
                    hidden_dim=64, lr=LR, gamma=0.99, gae_lambda=0.95,
                    clip_eps=0.2, coef_vf=0.5, coef_entropy=0.01,
                    n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, buffer_size=2048,
                    gat_hidden_dim=8, gat_n_heads=4, target_kl=TARGET_KL):

        self.agent_ids  = agent_ids
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.target_kl  = target_kl

        # PPO sur CPU (reseaux petits)
        self.ppo_device = torch.device("cpu")
        # GAT sur GPU si dispo, sinon CPU (4 noeuds : gain GPU minime en pratique)
        self.gat_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- GAT partage ----
        self.gat = GATModule(
            agent_order = agent_ids,
            obs_dim     = obs_dim,
            hidden_dim  = gat_hidden_dim,
            n_heads     = gat_n_heads,
            device      = self.gat_device
        )

        # ---- 4 agents PPO ----
        self.agents = {
            agent_id: PPOAgent(
                agent_id    = agent_id,
                obs_dim     = obs_dim,
                act_dim     = act_dim,
                hidden_dim  = hidden_dim,
                gamma       = gamma,
                gae_lambda  = gae_lambda,
                clip_eps    = clip_eps,
                coef_vf     = coef_vf,
                coef_entropy= coef_entropy,
                buffer_size = buffer_size,
                device      = self.ppo_device
            )
            for agent_id in agent_ids
        }

        # ---- UN SEUL optimizer sur GAT + tous les ActorCritic ----
        all_params = list(self.gat.network.parameters())
        for agent in self.agents.values():
            all_params += list(agent.network.parameters())

        self.optimizer = torch.optim.Adam(all_params, lr=lr)

        print(f"MultiAgentPPO+GAT initialise :")
        print(f"  PPO device : {self.ppo_device}")
        print(f"  GAT device : {self.gat_device}")
        print(f"  Agents : {list(self.agents.keys())}")
        print(f"  Target KL (early stopping) : {self.target_kl}")

    # ============================================================
    # SELECTION DES ACTIONS (jeu, pas de grad)
    # ============================================================

    def select_actions(self, obs_dict):
        """
        obs_dict : dict {agent_id: np.array(13,)} -- etats BRUTS (state.py)

        1. Enrichit les etats via GAT (no_grad, pour la vitesse)
        2. Chaque agent choisit son action depuis son etat enrichi
        3. On stocke aussi l'etat BRUT (necessaire pour re-enrichir a l'update)

        Retourne :
        actions, log_probs, values : dict {agent_id: ...}
        """
        enriched_dict, _ = self.gat.enrich(obs_dict, training=False)

        actions   = {}
        log_probs = {}
        values    = {}

        for agent_id, agent in self.agents.items():
            a, lp, v = agent.select_action(enriched_dict[agent_id])
            actions  [agent_id] = a
            log_probs[agent_id] = lp
            values   [agent_id] = v

        return actions, log_probs, values

    # ============================================================
    # STOCKAGE (on stocke l'etat BRUT, pas enrichi)
    # ============================================================

    def store(self, obs_brut_dict, actions, rewards, log_probs, values, done):
        for agent_id, agent in self.agents.items():
            agent.store(
                obs_brut = obs_brut_dict[agent_id],
                action   = actions  [agent_id],
                reward   = rewards  [agent_id],
                log_prob = log_probs[agent_id],
                value    = values   [agent_id],
                done     = float(done)
            )

    # ============================================================
    # UPDATE CONJOINT (GAT + 4 PPO, un seul backward)
    # ============================================================

    def update(self, last_obs_brut_dict=None):
        """
        Met a jour GAT + les 4 ActorCritic ENSEMBLE.

        Etapes :
        1. Verifie que tous les buffers sont prets
        2. Pour chaque epoch :
            a. Re-enrichit les etats bruts stockes via le GAT (AVEC grad)
            b. Calcule la loss PPO de chaque agent sur ses etats enrichis
               ainsi que la KL-divergence approximee par rapport a
               l'ancienne politique
            c. Si la KL moyenne (sur les 4 agents) depasse target_kl,
               on arrete les updates AVANT d'appliquer ce pas de gradient
               (early stopping) -- la politique et le GAT ont deja assez
               derive par rapport aux donnees collectees
            d. Sinon, on additionne les 4 loss -> loss_totale, backward,
               puis optimizer.step()
        3. Reset les buffers

        Retourne : dict {agent_id: loss moyenne, "_mean_kl": ..., "_stopped_early": ...}
                    ou None si pas pret
        """
        if not self.buffers_ready():
            return None

        # Valeur du dernier etat pour bootstrap GAE (par agent)
        last_values = {}
        if last_obs_brut_dict is not None:
            enriched_last, _ = self.gat.enrich(last_obs_brut_dict, training=False)
            for agent_id, agent in self.agents.items():
                obs_t = torch.FloatTensor(enriched_last[agent_id]).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    _, v = agent.network.forward(obs_t)
                last_values[agent_id] = v.item()
        else:
            last_values = {agent_id: 0.0 for agent_id in self.agents}

        n_steps = next(iter(self.agents.values())).buffer.ptr
        loss_history = {agent_id: [] for agent_id in self.agents}
        kl_history    = []
        stopped_early = False

        for epoch in range(self.n_epochs):
            # GAE recalcule une fois par epoch (pas par mini-batch), car il
            # doit etre calcule sur la trajectoire COMPLETE pour etre correct.
            for agent_id, agent in self.agents.items():
                agent.prepare_epoch(last_values[agent_id])

            indices = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.batch_size):
                idx = indices[start : start + self.batch_size]

                # ---- Reconstruire le batch d'etats bruts pour ce mini-batch ----
                obs_batch_raw = {}
                for agent_id, agent in self.agents.items():
                    obs_batch_raw[agent_id] = torch.FloatTensor(
                        agent.buffer.obs[idx]
                    ).to(self.gat_device)

                # ---- Re-enrichir via le GAT, AVEC grad ----
                enriched_batch, _ = self.gat.enrich_batch(obs_batch_raw)

                # ---- Calculer la loss et la KL de chaque agent ----
                total_loss     = 0.0
                kl_this_batch  = []
                for agent_id, agent in self.agents.items():
                        obs_enriched_for_agent = enriched_batch[agent_id].to(agent.device)

                        loss_agent, kl_agent = agent.compute_loss_minibatch(
                            obs_enriched_for_agent, idx, last_values[agent_id]
                        )
                        total_loss = total_loss + loss_agent
                        loss_history[agent_id].append(loss_agent.item())
                        kl_this_batch.append(kl_agent)

                    # Moyenne plutot que somme : le GAT est partage, donc sans
                    # cette division son gradient serait la somme des 4 agents
                    # (4x plus fort qu'attendu pour un LR pense pour un seul reseau)
                    total_loss = total_loss / len(self.agents)
                mean_kl = float(np.mean(kl_this_batch))
                kl_history.append(mean_kl)

                # ---- Early stopping : la politique/le GAT ont trop derive ----
                if mean_kl > self.target_kl:
                    print(f"  [Early stop] epoch {epoch+1}, KL={mean_kl:.4f} "
                            f"> target_kl={self.target_kl} -> update interrompue")
                    stopped_early = True
                    break

                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping sur TOUS les parametres (GAT + PPO)
                all_params = list(self.gat.network.parameters())
                for agent in self.agents.values():
                    all_params += list(agent.network.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                self.optimizer.step()

            if stopped_early:
                break

        # Reset des buffers
        for agent in self.agents.values():
            agent.reset_buffer()

        result = {
            agent_id: float(np.mean(losses)) if losses else 0.0
            for agent_id, losses in loss_history.items()
        }
        result["_mean_kl"]       = float(np.mean(kl_history)) if kl_history else 0.0
        result["_stopped_early"] = stopped_early

        return result

    # ============================================================
    # BUFFERS PRETS
    # ============================================================

    def buffers_ready(self):
        return all(agent.buffer.is_ready() for agent in self.agents.values())

    # ============================================================
    # SAUVEGARDE / CHARGEMENT
    # ============================================================

    def save(self, episode):
        save_dir = os.path.join(MODELS_DIR, f"episode_{episode}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(self.gat.network.state_dict(), os.path.join(save_dir, "gat.pt"))

        for agent_id, agent in self.agents.items():
            torch.save(agent.network.state_dict(), os.path.join(save_dir, f"{agent_id}.pt"))

        print(f"Modeles sauvegardes -> {save_dir}")

    def load(self, episode):
        load_dir = os.path.join(MODELS_DIR, f"episode_{episode}")

        self.gat.network.load_state_dict(
            torch.load(os.path.join(load_dir, "gat.pt"), map_location=self.gat_device)
        )
        for agent_id, agent in self.agents.items():
            agent.network.load_state_dict(
                torch.load(os.path.join(load_dir, f"{agent_id}.pt"), map_location=agent.device)
            )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(load_dir, "optimizer.pt"), map_location="cpu")
        )
        print(f"Modeles charges <- {load_dir}")