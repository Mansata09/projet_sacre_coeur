import os
import sys
import numpy as np

# Permet d'importer environnement/ et agents/ depuis training/
# en remontant a la racine du projet (sacre_coeur/)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environnement.traffic_env import TrafficEnv
from agents.multi_agent import MultiAgentPPO

# ============================================================
# CONFIGURATION ENTRAINEMENT
# ============================================================

N_EPISODES   = 20       # nombre total d'episodes
MAX_STEPS    = 3600     # duree d'un episode (= duree simulation SUMO)
BUFFER_SIZE  = 2048     # experiences avant un update PPO
SAVE_EVERY   = 20       # sauvegarder le modele toutes les N episodes
USE_GUI      = False    # mettre True pour voir SUMO en direct (plus lent)

LOG_DIR = os.path.join("..", "logs")

# ============================================================
# BOUCLE D'ENTRAINEMENT
# ============================================================

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "training_log.csv")

    env = TrafficEnv(max_steps=MAX_STEPS, use_gui=USE_GUI)
    multi_agent = MultiAgentPPO(buffer_size=BUFFER_SIZE)

    # En-tete du fichier de log
    with open(log_path, "w") as f:
        f.write("episode,step_total,reward_moyenne,reward_nord,reward_sud,reward_est,reward_ouest,accident,kl_moyenne,arret_anticipe\n")

    print("=" * 60)
    print("DEBUT ENTRAINEMENT")
    print(f"  Episodes      : {N_EPISODES}")
    print(f"  Max steps     : {MAX_STEPS}")
    print(f"  Buffer size   : {BUFFER_SIZE}")
    print("=" * 60)

    for episode in range(1, N_EPISODES + 1):

        obs_dict = env.reset(episode_num=episode)

        # Cumul des recompenses pour suivre la progression
        episode_rewards    = {agent_id: 0.0 for agent_id in multi_agent.agent_ids}
        accident_episode   = False
        last_obs_dict       = obs_dict
        last_kl             = 0.0
        last_stopped_early  = False

        for step in range(MAX_STEPS):

            # ---- 1. Selection des actions (via GAT + PPO) ----
            actions, log_probs, values = multi_agent.select_actions(obs_dict)

            # ---- 2. Avancer la simulation ----
            actions_list = [
                actions["agent_nord"],
                actions["agent_sud"],
                actions["agent_est"],
                actions["agent_ouest"]
            ]
            obs_next_dict, rewards, done, info = env.step(actions_list)

            # ---- 3. Stocker l'experience (etat BRUT, pas enrichi) ----
            multi_agent.store(obs_dict, actions, rewards, log_probs, values, done)

            # ---- 4. Cumuler les recompenses pour le log ----
            for agent_id in multi_agent.agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]

            if info["accident"]:
                accident_episode = True

            last_obs_dict = obs_next_dict
            obs_dict = obs_next_dict

            # ---- 5. Update PPO+GAT quand les buffers sont pleins ----
            if multi_agent.buffers_ready():
                losses = multi_agent.update(last_obs_brut_dict=obs_dict)
                if losses is not None:
                    # On separe les loss par agent des metadonnees (_mean_kl, _stopped_early)
                    loss_str = ", ".join(
                        f"{aid}={l:.4f}" for aid, l in losses.items()
                        if not aid.startswith("_")
                    )
                    last_kl            = losses["_mean_kl"]
                    last_stopped_early = losses["_stopped_early"]

                    kl_info = f"KL={last_kl:.4f}"
                    if last_stopped_early:
                        kl_info += " (early stop)"
                    print(f"  [Episode {episode}, step {step+1}] Update PPO+GAT -> "
                            f"{loss_str}, {kl_info}")

            if done:
                break

        # ---- Fin d'episode : forcer un dernier update si buffer partiel non vide ----
        # (optionnel : on choisit ici de NE PAS forcer, pour garder des batchs complets
        #  et un calcul GAE coherent ; les experiences restantes seront completees au
        #  prochain episode)

        # ---- Logging ----
        reward_moyenne = np.mean(list(episode_rewards.values()))
        print(f"Episode {episode}/{N_EPISODES} | "
                f"reward_moy={reward_moyenne:.4f} | "
                f"accident={accident_episode} | "
                f"steps={step+1}")

        with open(log_path, "a") as f:
            f.write(
                f"{episode},{step+1},{reward_moyenne:.6f},"
                f"{episode_rewards['agent_nord']:.6f},"
                f"{episode_rewards['agent_sud']:.6f},"
                f"{episode_rewards['agent_est']:.6f},"
                f"{episode_rewards['agent_ouest']:.6f},"
                f"{accident_episode},"
                f"{last_kl:.6f},"
                f"{last_stopped_early}\n"
            )

        # ---- Sauvegarde periodique ----
        if episode % SAVE_EVERY == 0:
            multi_agent.save(episode)

    # ---- Sauvegarde finale ----
    multi_agent.save("final")
    env.close()

    print("=" * 60)
    print("ENTRAINEMENT TERMINE")
    print(f"  Log -> {log_path}")
    print("=" * 60)


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    train()