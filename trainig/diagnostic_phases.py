"""
Script de diagnostic : verifie pour chaque phase du feu N0,
quel est l'etat (vert/orange/rouge) de chaque branche.

A lancer depuis le dossier training/ (ou n'importe ou avec le bon
sys.path vers environnement/).
"""

import os
import sys
sys.path.append('C:\\Program Files (x86)\\Eclipse\\Sumo\\tools')
import traci

SUMO_CFG = os.path.join('..', 'sumo_file', 'sacre_coeur.sumocfg')

BRANCH_TL_INDEX = {
    "agent_nord" : 1,
    "agent_sud"  : 11,
    "agent_est"  : 6,
    "agent_ouest": 17
}

def signal_to_label(signal):
    if signal in ('G', 'g'):
        return "VERT"
    elif signal in ('y', 'Y'):
        return "ORANGE"
    else:
        return "ROUGE"


if __name__ == "__main__":
    traci.start(["sumo", "-c", SUMO_CFG, "--no-warnings"])

    tls_id = "N0"
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)

    print("=" * 60)
    print(f"DIAGNOSTIC DES PHASES DU FEU {tls_id}")
    print("=" * 60)

    n_phases = len(logic[0].phases)
    print(f"\nNombre total de phases dans le programme : {n_phases}\n")

    for i, phase in enumerate(logic[0].phases):
        state_str = phase.state
        duration  = phase.duration

        print(f"--- Phase {i} (duree definie : {duration}s) ---")
        print(f"  string complet SUMO : {state_str}")

        for agent_id, index in BRANCH_TL_INDEX.items():
            if index < len(state_str):
                signal = state_str[index]
                label  = signal_to_label(signal)
                print(f"  {agent_id:12s} (index {index:2d}) : {signal} -> {label}")
            else:
                print(f"  {agent_id:12s} : INDEX HORS LIMITE (len={len(state_str)})")
        print()

    # ---- Resume : est-ce que chaque agent a au moins UN vert ? ----
    print("=" * 60)
    print("RESUME : chaque branche a-t-elle un VERT dans le cycle ?")
    print("=" * 60)

    for agent_id, index in BRANCH_TL_INDEX.items():
        has_green = False
        for phase in logic[0].phases:
            if index < len(phase.state) and phase.state[index] in ('G', 'g'):
                has_green = True
                break
        status = "OUI" if has_green else "NON <-- PROBLEME !"
        print(f"  {agent_id:12s} : {status}")

    traci.close()
    print("\nDiagnostic termine.")