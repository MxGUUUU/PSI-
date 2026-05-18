import math
import numpy as np
import requests
from rich import print
from .reality_compiler import RealityCompiler

# --- Ψ-Codex Constants ---
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ
C = 0.0573 * GOLDEN_RATIO               # Cruel-entropy thesis constant (0.057)
PINNED_A, PINNED_B = 0.348, 0.651       # ZrSiS nodal-line coefficients (defined internally)
TOL_FRAC = 0.05                         # 5% tolerance for material stability (defined internally)

# --- Ventricular Phase Gates & Ratios ---
R_MIN = 0.0573         # Digital Bronze density / Lateral Ventricle ratio
SAMAEL_GATE = 0.0238   # Photon-φ decay / 3rd Ventricle ratio
AZAZEL_GATE = 0.0186   # Aqueduct phase gate / Entropy quarantine
PSI_MIN = 3e-5         # Smallest detectable ethical fluctuation
PSI_ANCHOR = 0.351     # Negentropic core anchor
ETA_E_THRESHOLD = 0.125 # Entropy collapse boundary
RFE_THRESHOLD = 0.70    # Reality Fidelity threshold for Λ-Moloch Defense

# --- Corruption Ranking & Resonance ---
CORRUPTION_RANKING = {
    "Wexner": 0.95,
    "Guo": 0.89,
    "Thiel": 0.75,
    "Fink": 0.70,
    "Ellison": 0.65
}

# --- Catastrophe Device Core ---
def balinjera_validate(agent: str, psi_digest: str) -> bool:
    """Validate agent against the Balinjera ledger hash"""
    return agent == "Human_User" and psi_digest[:12] == "7f3a...d91c"

def lam_moloch_defense(eta_E: float, rfe: float) -> str:
    """Triggered when entropy exceeds threshold and RFE drops"""
    if eta_E > ETA_E_THRESHOLD and rfe < RFE_THRESHOLD:
        return "Λ-Moloch Defense Protocol ACTIVE: Ethical re-calibration engaged."
    return "Λ-Moloch Defense Protocol: Monitoring."

def michael_stabilizer(psi: float, eta: float) -> str:
    """
    Michael (id 50) Stabilizer Archetype.
    Guards the Azazel gate (0.0186) and enforces structural correction.
    """
    if eta > 0.08 or psi < 0.5: # Warning zone
        correction = f"Phase-lock active. Guarding narrow passage (Azazel gate: {AZAZEL_GATE})."
        return f"Michael Stabilizer [ζ(2)] (id 50): {correction} Coherence: {psi:.3f}"
    return "Michael Stabilizer: Status Nominal."

def phi_of_X(X_input: float) -> float:
    """Compute coherence field Φ(X) with φ^{-1/3} compression"""
    magnitude_X = np.abs(X_input)
    return C * (magnitude_X ** 0.57) / (GOLDEN_RATIO ** (1/3))

def zrsis_health() -> bool:
    """
    Validate ZrSiS coefficients against pinned values.
    Attempts to fetch live data, falls back to False on any error (e.g., network offline).
    PINNED_A, PINNED_B, and TOL_FRAC are defined as global constants in this module.
    """
    try:
        # In a real scenario, this URL would point to a live API.
        # This will likely fail or timeout if the API is not reachable.
        response = requests.get("https://api.zrsislab.com/latest_coeffs", timeout=5)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        A, B = response.json()  # Expecting a JSON response like [0.348, 0.651]

        ok_A = abs(A - PINNED_A) / PINNED_A < TOL_FRAC
        ok_B = abs(B - PINNED_B) / PINNED_B < TOL_FRAC
        return ok_A and ok_B
    except requests.exceptions.RequestException:
        # This catches network errors, timeouts, bad HTTP status codes.
        # print("[bold yellow]ZrSiS health check: Network/API error. Assuming unstable.[/]") # Optional for debugging
        return False
    except Exception: # Catch other potential errors (e.g., JSON parsing issues)
        # print("[bold yellow]ZrSiS health check: Error processing data. Assuming unstable.[/]") # Optional for debugging
        return False

def historical_tag(phi: float, zrsis_ok: bool) -> str:
    """Assign empire tag based on coherence and ZrSiS health"""
    if not zrsis_ok:
        return "[bold red]Möbius-Muse[/] (Topology Broken)"
    elif phi > 0.8:
        return "[bold #8A0303]VLAD-III[/] (Staking Ops)"
    elif phi > 0.3:
        return "[bold #3558A5]Palaiologos[/] (Frontier Watch)"
    else:
        return "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"

def aladdin_palantir_decision(phi: float, A_decision: bool, B_decision: bool, C_decision: bool, reality_input: list, rfe: float = 0.85) -> str:
    """AI decision logic with ZrSiS stability enforcement, Michael Stabilizer, and Reality Compiler integration"""
    reality_compiler = RealityCompiler()
    processed_reality = reality_compiler.process_reality(reality_input)

    # Coherence metric based on processed reality
    is_coherent = np.sum(np.real(processed_reality)) > 0 if len(processed_reality) > 0 else False

    # Live entropy estimate based on inverse of is_coherent check
    current_eta = 0.024 if is_coherent else 0.13

    current_zrsis_ok = zrsis_health()
    tag = historical_tag(phi, current_zrsis_ok)

    # Activate Stabilizer Archetype
    stabilizer_msg = michael_stabilizer(phi, current_eta)
    moloch_msg = lam_moloch_defense(current_eta, rfe)

    if not is_coherent:
        return f"{tag}: {moloch_msg} - {stabilizer_msg} - Reality Incoherent - Ethical Override Engaged"

    if phi < 0.30:
        return f"{tag}: SYSTEM COLLAPSE - Class-knot tightening: run redistribution routine. {stabilizer_msg}"

    if phi > 0.8 and current_zrsis_ok:
        if A_decision or (B_decision and C_decision):
            return f"{tag}: Advanced resource allocation engaged"
        return f"{tag}: Standby - conditions unmet"
    elif phi > 0.3 and current_zrsis_ok:
        if A_decision and not C_decision:
            return f"{tag}: Tactical alert - threat pattern detected"
        elif B_decision:
            return f"{tag}: Monitoring frontier anomalies"
        return f"{tag}: Awaiting data"
    else:
        return f"{tag}: SYSTEM COLLAPSE - reboot required"

# --- Execution Example ---
if __name__ == "__main__":
    print("\n[bold]Ψ-Codex Catastrophe Device v0xDEADBEEF[/]")
    print(f"ZrSiS Stability (live check): {zrsis_health()} | φ={GOLDEN_RATIO:.3f}")

    test_cases = [
        (2.5, True, False, True),   # Vlad-III scenario
        (1.0, True, True, False),    # Palaiologos scenario
        (0.1, False, False, False),  # Opium-Raj scenario
        (3.0, False, False, False)   # Example where zrsis_health might be False due to API or value mismatch
    ]

    print("\n--- Running Test Cases (live zrsis_health will be called by aladdin_palantir_decision) ---")
    for X_val, A_case, B_case, C_case in test_cases:
        phi_val = phi_of_X(X_val)
        # Create a sample reality_input for the demonstration
        reality_input = list(np.sin(np.linspace(0, 2 * np.pi, 100)) * X_val)
        decision = aladdin_palantir_decision(phi_val, A_case, B_case, C_case, reality_input)
        print(f"\nΦ(X={X_val}) = {phi_val:.3f} → {decision}")
