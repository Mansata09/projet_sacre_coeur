import numpy as np
import sys
sys.path.append('C:\\Program Files (x86)\\Eclipse\\Sumo\\tools')
import traci

# ============================================================
# CONFIGURATION
# ============================================================

ACTION_NAMES = {
    0: "garder",
    1: "changer",
    2: "prolonger_5s",
    3: "prolonger_10s"
}

MIN_GREEN    = 10   # durée minimum vert (secondes)
PROLONG_5S   = 5    # prolongation courte
PROLONG_10S  = 10   # prolongation longue

# ============================================================
# VALIDATION DES ACTIONS
# ============================================================

def normalize_actions(actions, n_agents=4):
    """Vérifie qu'on a bien 4 actions valides"""
    arr = np.asarray(actions, dtype=int).reshape(-1)
    if arr.size != n_agents:
        raise ValueError(f"Attendu : {n_agents} actions, reçu : {arr.size}")
    if np.any((arr < 0) | (arr > 3)):
        raise ValueError("Les actions doivent être entre 0 et 3")
    return arr

# ============================================================
# APPLICATION DES ACTIONS
# ============================================================

def apply_actions(actions, tls_id="N0", phase_timer=0):
    """
    Applique les actions des 4 agents au feu de N0.
    
    Logique de priorité :
    1. Ambulance en attente → forcer changement
    2. BRT en attente → prolonger ou changer selon contexte
    3. Majorité des agents → décision collective
    
    actions : liste de 4 actions [nord, sud, est, ouest]
    tls_id  : id du feu dans SUMO
    phase_timer : temps écoulé dans la phase actuelle
    """
    actions = normalize_actions(actions)
    act_nord, act_sud, act_est, act_ouest = actions

    try:
        # ---- PRIORITE ABSOLUE : Ambulance ----
        for veh in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh) == "emergency":
                    if traci.vehicle.getWaitingTime(veh) > 3:
                        # Forcer changement de phase
                        _changer_phase(tls_id, phase_timer)
                        return "changer (priorité ambulance)"
            except:
                pass

        # ---- PRIORITE HAUTE : BRT en attente ----
        for veh in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh) == "brt":
                    if traci.vehicle.getWaitingTime(veh) > 5:
                        _changer_phase(tls_id, phase_timer)
                        return "changer (priorité BRT)"
            except:
                pass

        # ---- DECISION COLLECTIVE des 4 agents ----
        # Compter les votes
        votes = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in actions:
            votes[a] += 1

        # Action avec le plus de votes
        action_choisie = max(votes, key=votes.get)

        if action_choisie == 0:
            # Garder
            return "garder"

        elif action_choisie == 1:
            # Changer (si temps minimum respecté)
            if phase_timer >= MIN_GREEN:
                _changer_phase(tls_id, phase_timer)
                return "changer"
            else:
                return "garder (temps minimum non respecté)"

        elif action_choisie == 2:
            # Prolonger +5s
            if phase_timer >= MIN_GREEN:
                _prolonger_phase(tls_id, PROLONG_5S)
                return "prolonger_5s"
            else:
                return "garder"

        elif action_choisie == 3:
            # Prolonger +10s
            if phase_timer >= MIN_GREEN:
                _prolonger_phase(tls_id, PROLONG_10S)
                return "prolonger_10s"
            else:
                return "garder"

    except Exception as e:
        print(f"Erreur apply_actions : {e}")
        return "garder"

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def _changer_phase(tls_id, phase_timer):
    """Valide juste que le changement est possible"""
    try:
        if phase_timer >= MIN_GREEN:
            return True  # traffic.py s'occupe du changement
        return False
    except:
        return False
def _prolonger_phase(tls_id, duree):
    """Prolonge la phase actuelle de N secondes"""
    try:
        current_duration = traci.trafficlight.getPhaseDuration(tls_id)
        traci.trafficlight.setPhaseDuration(tls_id, current_duration + duree)
    except Exception as e:
        print(f"Erreur prolongation : {e}")
