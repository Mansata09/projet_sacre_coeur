import numpy as np
import sys
sys.path.append('C:\\Program Files (x86)\\Eclipse\\Sumo\\tools')
import traci

# ============================================================
# CONFIGURATION DES VOIES PAR BRANCHE
# ============================================================

LANES_NORD  = ["N3_N0_0", "N3_N0_BRT_0"]  # Liberté 6 → avec BRT
LANES_SUD   = ["N4_N0_0", "N4_N0_BRT_0"]  # Petersen → avec BRT
LANES_EST   = ["N2_N0_0"]                  # Sacré-Cœur → sans BRT
LANES_OUEST = ["N1_N0_0"]                  # Ouakam → sans BRT

# Index des feux par branche dans le string SUMO
BRANCH_TL_INDEX = {
    "agent_nord" : 1,   # N3_N0
    "agent_sud"  : 11,  # N4_N0
    "agent_est"  : 6,   # N2_N0
    "agent_ouest": 17   # N1_N0
}

SIM_DURATION  = 3600.0
MAX_QUEUE     = 50.0
MAX_WAIT      = 120.0
MAX_OCCUPANCY = 100.0
MAX_PHASE     = 6.0
MAX_PHASE_DUR = 60.0

# Périodes (en secondes)
PERIOD_MATIN_END  = 1000.0
PERIOD_CREUSE_END = 2000.0
PERIOD_SOIR_END   = 3000.0

# ============================================================
# ETAT DU FEU PAR BRANCHE
# ============================================================

def get_feu_state(tls_id, agent_id):
    """
    Retourne l'état du feu pour une branche :
    1.0 = vert
    0.5 = orange
    0.0 = rouge
    """
    try:
        state  = traci.trafficlight.getRedYellowGreenState(tls_id)
        index  = BRANCH_TL_INDEX[agent_id]
        signal = state[index]

        if signal in ('G', 'g'):
            return 1.0
        elif signal in ('y', 'Y'):
            return 0.5
        else:
            return 0.0
    except:
        return 0.0

# ============================================================
# FEATURES PAR BRANCHE
# ============================================================
def get_branch_metrics(lane_ids, check_brt=False):
    """
    Calcule 7 features pour une branche :
    1. File d'attente
    2. Temps d'attente moyen PAR VEHICULE
    3. Occupation
    4. BRT en approche (0/1)
    5. BRT en attente (0/1)
    6. Ambulance en approche (0/1)
    7. Ambulance en attente (0/1)
    """
    total_queue     = 0
    total_wait      = 0
    total_occupancy = 0
    total_vehicles  = 0
    brt_approche    = 0.0
    brt_attente     = 0.0
    amb_approche    = 0.0
    amb_attente     = 0.0

    for lane in lane_ids:
        try:
            total_queue     += traci.lane.getLastStepHaltingNumber(lane)
            total_wait      += traci.lane.getWaitingTime(lane)
            total_occupancy += traci.lane.getLastStepOccupancy(lane)

            vehicle_ids     = traci.lane.getLastStepVehicleIDs(lane)
            total_vehicles += len(vehicle_ids)

            for veh_id in vehicle_ids:
                try:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    speed    = traci.vehicle.getSpeed(veh_id)

                    if veh_type == "emergency":
                        if speed < 0.1:
                            amb_attente  = 1.0
                        else:
                            amb_approche = 1.0

                    if check_brt and veh_type == "brt":
                        if speed < 0.1:
                            brt_attente  = 1.0
                        else:
                            brt_approche = 1.0
                except:
                    pass
        except:
            pass

    n          = max(len(lane_ids), 1)
    avg_queue  = min(total_queue     / n / MAX_QUEUE,     1.0)
    avg_occ    = min(total_occupancy / n / MAX_OCCUPANCY, 1.0)

    # Temps d'attente moyen PAR VEHICULE (et non plus par voie)
    # Si aucun vehicule present, le temps d'attente moyen est 0
    n_veh    = max(total_vehicles, 1) if total_vehicles > 0 else 1
    avg_wait = min(total_wait / n_veh / MAX_WAIT, 1.0) if total_vehicles > 0 else 0.0

    return [
        avg_queue,    # 1. File d'attente
        avg_wait,     # 2. Temps d'attente par vehicule
        avg_occ,      # 3. Occupation
        brt_approche, # 4. BRT en approche
        brt_attente,  # 5. BRT en attente
        amb_approche, # 6. Ambulance en approche
        amb_attente   # 7. Ambulance en attente
    ]
# ============================================================
# CONSTRUCTION DE L'ETAT MULTI-AGENT
# ============================================================

def build_multi_agent_state(tls_id="N0"):
    """
    Chaque agent reçoit 13 features :
    1-7  : features de sa branche
    8    : état du feu (0=rouge, 0.5=orange, 1=vert)
    9    : phase normalisée
    10   : durée phase normalisée
    11   : heure normalisée
    12   : is_pointe_matin (1.0 si 0-1000s, sinon 0.0)
    13   : is_pointe_soir  (1.0 si 2000-3000s, sinon 0.0)
    """
    try:
        current_phase  = traci.trafficlight.getPhase(tls_id)
        next_switch    = traci.trafficlight.getNextSwitch(tls_id)
        sim_time       = traci.simulation.getTime()
        phase_duration = max(next_switch - sim_time, 0.0)
        phase_norm     = current_phase  / MAX_PHASE
        duration_norm  = min(phase_duration / MAX_PHASE_DUR, 1.0)
        heure_norm     = sim_time / SIM_DURATION
    except:
        phase_norm = duration_norm = heure_norm = 0.0
        sim_time   = 0.0

    # Périodes
    is_pointe_matin = 1.0 if sim_time < PERIOD_MATIN_END else 0.0
    is_pointe_soir  = 1.0 if PERIOD_CREUSE_END <= sim_time < PERIOD_SOIR_END else 0.0

    # Features par branche
    features_nord  = get_branch_metrics(LANES_NORD,  check_brt=True)
    features_sud   = get_branch_metrics(LANES_SUD,   check_brt=True)
    features_est   = get_branch_metrics(LANES_EST,   check_brt=False)
    features_ouest = get_branch_metrics(LANES_OUEST, check_brt=False)

    # Etat du feu par branche
    feu_nord  = get_feu_state(tls_id, "agent_nord")
    feu_sud   = get_feu_state(tls_id, "agent_sud")
    feu_est   = get_feu_state(tls_id, "agent_est")
    feu_ouest = get_feu_state(tls_id, "agent_ouest")

    # Contexte global (5 features)
    contexte = [phase_norm, duration_norm, heure_norm, is_pointe_matin, is_pointe_soir]

    return {
        "agent_nord" : np.array(features_nord  + [feu_nord]  + contexte, dtype=np.float32),
        "agent_sud"  : np.array(features_sud   + [feu_sud]   + contexte, dtype=np.float32),
        "agent_est"  : np.array(features_est   + [feu_est]   + contexte, dtype=np.float32),
        "agent_ouest": np.array(features_ouest + [feu_ouest] + contexte, dtype=np.float32)
    }

#