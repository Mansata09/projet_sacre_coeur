import numpy as np
import sys
sys.path.append('C:\\Program Files (x86)\\Eclipse\\Sumo\\tools')
import traci

# ============================================================
# COEFFICIENTS DE PENALITE
# ============================================================

DEFAULT_WEIGHTS = {
    "waiting_voitures"   : 1.0,     # penalite file attente
    "density"            : 10.0,    # penalite densite
    "brt_waiting_factor" : 2.0,     # penalite BRT en attente
    "ambulance_waiting"  : 500.0,   # penalite ambulance bloquee
    "ambulance_approche" : 100.0,   # penalite ambulance en approche
    "accident"           : 1000.0,  # penalite accident
    "stability"          : 30.0,    # penalite changement trop rapide
    "balance"            : 10.0     # penalite desequilibre reseau
}

# ============================================================
# PENALITE LOCALE PAR BRANCHE
# ============================================================

def compute_branch_local_reward(lane_ids, check_brt=False, weights=None):
    """
    Calcule la penalite locale pour UNE branche.
    
    Retourne une valeur positive (sera mise en negatif apres).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    total_queue    = 0
    total_occupancy= 0
    brt_penalty    = 0.0
    ambulance_penalty = 0.0

    for lane in lane_ids:
        try:
            total_queue     += traci.lane.getLastStepHaltingNumber(lane)
            total_occupancy += traci.lane.getLastStepOccupancy(lane)

            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                try:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    speed    = traci.vehicle.getSpeed(veh_id)

                    # ---- Ambulance ----
                    if veh_type == "emergency":
                        if speed < 0.1:
                            # Ambulance bloquee → penalite maximale
                            wait = traci.vehicle.getWaitingTime(veh_id)
                            ambulance_penalty += w["ambulance_waiting"] * (1 + wait / 3.0)
                        else:
                            # Ambulance en approche → penalite moderee
                            ambulance_penalty += w["ambulance_approche"]

                    # ---- BRT (seulement axes Nord/Sud) ----
                    if check_brt and veh_type == "brt":
                        if speed < 0.1:
                            wait = traci.vehicle.getWaitingTime(veh_id)
                            # Penalite quadratique normalisee
                            brt_penalty += w["brt_waiting_factor"] * (wait / 30.0) ** 2

                except:
                    pass

        except:
            pass

    # Penalite trafic general
    traffic_penalty = (
        total_queue     * w["waiting_voitures"] +
        total_occupancy * w["density"]
    )

    return traffic_penalty + brt_penalty + ambulance_penalty


# ============================================================
# RECOMPENSE TOTALE MULTI-AGENT
# ============================================================

def compute_total_rewards(lane_mappings, last_phase, current_phase,
                        phase_duration, weights=None):
    """
    Calcule les recompenses pour les 4 agents.

    Parametres :
    - lane_mappings  : dict {agent_id: [lane_ids]}
    - last_phase     : phase precedente du feu
    - current_phase  : phase actuelle du feu
    - phase_duration : duree ecoulee dans la phase actuelle

    Retourne :
    - rewards          : dict {agent_id: float}
    - collision_detected : bool
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # ---- 1. Accident (collision) ----
    try:
        collisions        = traci.simulation.getCollidingVehiclesIDList()
        collision_detected = len(collisions) > 0
        collision_penalty  = w["accident"] if collision_detected else 0.0
    except:
        collision_detected = False
        collision_penalty  = 0.0

    # ---- 2. Stabilite (changement trop rapide) ----
    # NOTE : cette penalite se redeclenche a CHAQUE step tant que
    # phase_duration < MIN_STABLE_DURATION, pas seulement au moment du
    # changement. On la limite donc a un seuil plus bas (5s) pour ne pas
    # noyer le signal avec un bruit constant de +30 a chaque step
    # pendant les 10 premieres secondes de CHAQUE phase.
    MIN_STABLE_DURATION = 5.0
    stability_penalty = 0.0
    if last_phase != current_phase and phase_duration < MIN_STABLE_DURATION:
        stability_penalty = w["stability"]

    # ---- 3. Penalites locales par branche ----
    local_penalties = {}
    for agent_id, lanes in lane_mappings.items():
        is_brt = agent_id in ["agent_nord", "agent_sud"]
        local_penalties[agent_id] = compute_branch_local_reward(
            lanes, check_brt=is_brt, weights=w
        )

    # ---- 4. Balance (equilibre entre branches) ----
    # IMPORTANT : balance_penalty est une grandeur GLOBALE (calculee une
    # seule fois pour tout le carrefour), mais elle etait avant ajoutee
    # INTEGRALEMENT a CHACUN des 4 agents. Resultat : les 4 agents recevaient
    # quasi la meme grosse valeur a chaque step, ce qui dominait le signal
    # individuel et faisait exploser la loss du Critic (les 4 agents
    # "voyaient" presque le meme reward, rendant le signal peu informatif
    # et de trop grande amplitude). On la divise par n_agents pour qu'elle
    # reste une correction d'appoint, pas le terme dominant.
    n_agents        = len(local_penalties)
    balance_index   = float(np.std(list(local_penalties.values())))
    balance_penalty = (balance_index * w["balance"]) / n_agents

    # ---- 5. Recompense finale par agent ----
    rewards = {}
    for agent_id in lane_mappings.keys():
        raw_reward = -(
            local_penalties[agent_id] +
            balance_penalty           +
            collision_penalty         +
            stability_penalty
        )
        # Normalisation pour PPO
        rewards[agent_id] = raw_reward / 1000.0

    return rewards, collision_detected


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    import os

    SUMO_CFG = os.path.join('..', 'sumo_file', 'sacre_coeur.sumocfg')

    LANE_MAPPINGS = {
        "agent_nord" : ["N3_N0_0", "N3_N0_BRT_0"],
        "agent_sud"  : ["N4_N0_0", "N4_N0_BRT_0"],
        "agent_est"  : ["N2_N0_0"],
        "agent_ouest": ["N1_N0_0"]
    }

    traci.start(["sumo", "-c", SUMO_CFG, "--no-warnings"])
    traci.simulationStep()

    last_phase    = 0
    current_phase = traci.trafficlight.getPhase("N0")
    phase_duration= 5.0

    rewards, collision = compute_total_rewards(
        LANE_MAPPINGS, last_phase, current_phase, phase_duration
    )

    print("\n=== RECOMPENSES ===")
    for agent, r in rewards.items():
        print(f"{agent} : {r:.6f}")
    print(f"Accident détecté : {collision}")

    traci.close()
    print("\nReward OK !")