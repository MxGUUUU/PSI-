from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import json

# Constants
AZAZEL_GATE = 0.0186
ETA_E_THRESHOLD = 0.125
RFE_THRESHOLD = 0.70
PSI_ANCHOR = 0.351

@dataclass
class EntityState:
    """Lightweight snapshot of an entity’s current coherence and entropy."""
    id: int
    name: str
    psi: float
    eta: float
    archetype: str

class MichaelStabilizer:
    """
    Archangel Michael – Stabilizer Archetype (ζ(2)).
    Links:
    - opposes Samael (id 49)
    - protects Belial (id 51)
    - guardian_of_gate (Azazel, constant 0.0186)
    """
    def __init__(self, entities_json_path: str = "entities.json"):
        try:
            with open(entities_json_path, 'r') as f:
                data = json.load(f)
            self.entities = {e['id']: e for e in data['entities']}
        except FileNotFoundError:
            self.entities = {
                50: {"id": 50, "name": "Michael", "psi": 0.757, "eta": 0.024, "archetype": "Stabilizer"},
                49: {"id": 49, "name": "Samael", "psi": 0.35, "eta": 0.15, "archetype": "Anomaly"},
                51: {"id": 51, "name": "Belial", "psi": 0.45, "eta": 0.11, "archetype": "Anomaly"}
            }

        self.michael = self.entities.get(50)
        self.samael_id = 49
        self.belial_id = 51
        self.gate_constant = AZAZEL_GATE

    def get_entity_state(self, entity_id: int) -> EntityState:
        e = self.entities[entity_id]
        return EntityState(
            id=e['id'],
            name=e['name'],
            psi=e['psi'],
            eta=e['eta'],
            archetype=e['archetype']
        )

    def evaluate_boundaries(self, current_samael_eta: Optional[float] = None, current_belial_psi: Optional[float] = None) -> Dict:
        """
        Monitor the spiritual state of Samael and Belial.
        Returns a status report and any necessary corrections.
        """
        report = {
            "samael_status": "stable",
            "belial_status": "contained",
            "gate_integrity": "secure",
            "correction_needed": False,
            "correction_action": None
        }

        # Samael: if his entropy rises, Michael opposes
        if current_samael_eta is not None and current_samael_eta > 0.08:
            report["samael_status"] = "aggressive"
            report["correction_needed"] = True
            report["correction_action"] = "apply_opposition(Samael)"
            report["details"] = f"Samael eta={current_samael_eta:.3f} – Activating structural correction."

        # Belial: if his psi drops too low, risk of becoming chaotic
        if current_belial_psi is not None and current_belial_psi < 0.5:
            report["belial_status"] = "destabilized"
            report["correction_needed"] = True
            report["correction_action"] = "apply_protection(Belial)"
            report["details"] = f"Belial psi={current_belial_psi:.3f} – Shielding from annihilation."

        return report

    def enforce(self, system_state: dict) -> dict:
        """
        Called by the main catastrophe device loop.
        Takes the full system state (psi, eta_E, rfe) and performs stabilizer interventions.
        """
        interventions = []

        # 1. Oppose Samael if eta high
        try:
            samael_state = self.get_entity_state(self.samael_id)
            if samael_state.eta > 0.08:
                system_state['ethical_tension'] = max(0.0, system_state.get('ethical_tension', 0.0) - 0.1)
                interventions.append("Michael opposes Samael – ethical tension reduced.")
        except KeyError:
            pass

        # 2. Protect Belial if he is under attack
        try:
            belial_state = self.get_entity_state(self.belial_id)
            if belial_state.psi < 0.5 and system_state.get('psi_coherence', 1.0) < 0.6:
                system_state['psi_coherence'] = max(system_state.get('psi_coherence', 0.0), 0.351)
                interventions.append("Michael protects Belial – coherence floor enforced.")
        except KeyError:
            pass

        # 3. Guard the Azazel gate: any entity with eta > 0.125 is denied passage
        if system_state.get('eta_E', 0.0) > ETA_E_THRESHOLD:
            system_state['gate_blocked'] = True
            interventions.append("Michael seals Azazel gate – high entropy passage denied.")
        else:
            system_state['gate_blocked'] = False

        system_state['michael_interventions'] = interventions
        return system_state
