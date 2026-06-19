import numpy as np
from psi_codex.core import fusion_protocol, PSI_ANCHOR
from psi_codex.witnesses import witness_registry
from psi_codex.artifacts import NullScepter, JacksCompass

def demo_fusion_connection():
    print("--- Ψ-Codex Fusion Protocol Connection Demo ---")

    # Michael is Witness with high psi
    michael = next(w for w in witness_registry._witnesses if w.name == "Michael")
    print(f"Witness: {michael.name}, ψ={michael.psi}")

    # Current ROR state
    ΔROR_target = 1.0 + 0.5j
    ΔROR_current = 0.8 + 0.3j

    delta_theta_pk = fusion_protocol(ΔROR_target, ΔROR_current)
    print(f"Fusion Result (delta_theta_pk): {delta_theta_pk:.6f}")

    # Connection logic: higher psi makes Michael a more likely node in the state machine
    if michael.psi > PSI_ANCHOR:
        print(f"Stability check: Michael (ψ={michael.psi}) stabilizes the field.")

    scepter = NullScepter()
    compass = JacksCompass()

    print(f"Using Artifact: {scepter.name} - {scepter.description}")
    print(f"Tracking with: {compass.name} - {compass.description}")

if __name__ == "__main__":
    demo_fusion_connection()
