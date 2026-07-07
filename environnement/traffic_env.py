import os
import sys
import numpy as np
sys.path.append('C:\\Program Files (x86)\\Eclipse\\Sumo\\tools')
import traci

# APRES
from environnement.state import build_multi_agent_state
from environnement.action_mapper import apply_actions, ACTION_NAMES
from environnement.reward import compute_total_rewards
# ============================================================
# CONFIGURATION
# ============================================================

SUMO_CFG = os.path.join('..', 'sumo_file', 'sacre_coeur.sumocfg')

LANE_MAPPINGS = {
    "agent_nord" : ["N3_N0_0", "N3_N0_BRT_0"],
    "agent_sud"  : ["N4_N0_0", "N4_N0_BRT_0"],
    "agent_est"  : ["N2_N0_0"],
    "agent_ouest": ["N1_N0_0"]
}

# ============================================================
# ENVIRONNEMENT
# ============================================================

class TrafficEnv:

    def __init__(self, max_steps=3600, use_gui=False):
        self.max_steps      = max_steps
        self.use_gui        = use_gui
        self.tls_id         = "N0"
        self.current_step   = 0
        self.last_phase     = 0
        self.current_phase  = 0
        self.phase_duration = 0.0
        self.sumo_running   = False

        # Transition feu jaune
        self.in_yellow      = False
        self.yellow_timer   = 0
        self.target_phase   = 0

    def reset(self, episode_num=None):
        """
        Redemarre la simulation SUMO.

        episode_num : si fourni, utilise ce numero comme seed pour que
        chaque episode genere un trafic DIFFERENT (flows en probability=).
        Sans seed explicite, SUMO reutilise toujours la meme sequence
        aleatoire par defaut, ce qui produit des episodes identiques.
        """
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass

        binary = "sumo-gui" if self.use_gui else "sumo"

        # Seed variable : basee sur l'episode si fourni, sinon aleatoire
        if episode_num is not None:
            seed = episode_num
        else:
            seed = np.random.randint(0, 100000)

        traci.start([
            binary, "-c", SUMO_CFG,
            "--no-warnings",
            "--seed", str(seed)
        ])

        self.sumo_running   = True
        self.current_step   = 0
        self.phase_duration = 0.0
        self.in_yellow      = False
        self.yellow_timer   = 0

        traci.simulationStep()

        self.last_phase    = traci.trafficlight.getPhase(self.tls_id)
        self.current_phase = self.last_phase

        return build_multi_agent_state(self.tls_id)

    def step(self, actions):
        """
        Execute un pas de simulation.
        actions : liste de 4 actions [nord, sud, est, ouest]
        """
        # ---- Gestion transition jaune ----
        if self.in_yellow:
            self.yellow_timer -= 1
            if self.yellow_timer <= 0:
                self.in_yellow = False
                traci.trafficlight.setPhase(self.tls_id, self.target_phase)
                self.last_phase    = self.current_phase
                self.current_phase = self.target_phase
                self.phase_duration = 0.0
        else:
            # Appliquer les actions
            resultat = apply_actions(
                actions,
                tls_id      = self.tls_id,
                phase_timer = self.phase_duration
            )

            # Si changement de phase → passer par jaune
            if "changer" in resultat:
                try:
                    logic    = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
                    n_phases = len(logic[0].phases)
                    next_p   = (self.current_phase + 1) % n_phases
                    # Phase jaune = phase actuelle + 1
                    yellow_p = (self.current_phase + 1) % n_phases
                    traci.trafficlight.setPhase(self.tls_id, yellow_p)
                    self.in_yellow    = True
                    self.yellow_timer = 3
                    self.target_phase = (yellow_p + 1) % n_phases
                except:
                    pass

        # ---- Avancer simulation ----
        traci.simulationStep()
        self.current_step   += 1
        self.phase_duration += 1.0

        # ---- Collecter état ----
        obs = build_multi_agent_state(self.tls_id)

        # ---- Calculer récompenses ----
        rewards, accident = compute_total_rewards(
            LANE_MAPPINGS,
            self.last_phase,
            self.current_phase,
            self.phase_duration
        )

        # ---- DEBUG TEMPORAIRE : detecter les recompenses anormales ----
        for agent_id, r in rewards.items():
            if abs(r) > 5.0:
                print(f"REWARD ANORMAL: {agent_id} = {r:.2f}  step={self.current_step}  phase_dur={self.phase_duration}")

        # ---- Condition d'arrêt ----
        done = (self.current_step >= self.max_steps) or accident

        info = {
            "step"        : self.current_step,
            "accident"    : accident,
            "phase"       : self.current_phase,
            "in_yellow"   : self.in_yellow
        }

        return obs, rewards, done, info

    def close(self):
        """Ferme la simulation"""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False