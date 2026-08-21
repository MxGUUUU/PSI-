import math
import numpy as np
import requests
from rich import print
from .reality_compiler import RealityCompiler

# --- Ψ-Codex Core Constants ---
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ = 1.618...
GOLDEN_ANGLE = 137.507764               # Golden angle in degrees
C = 0.0573 * GOLDEN_RATIO               # Cruel-entropy thesis constant
PINNED_A, PINNED_B = 0.348, 0.651       # ZrSiS nodal-line coefficients
TOL_FRAC = 0.05                         # 5% tolerance
PSI_ANCHOR = 0.351                      # Negentropic core anchor (coherence floor)
ETA_E_THRESHOLD = 0.125                 # Entropy collapse boundary
RFE_THRESHOLD = 0.70                    # Reality Fidelity threshold
AZAZEL_GATE = 0.0186                    # Aqueduct phase gate
LAMBDA_3 = 1.1                          # Single canonical definition of resilience factor λ₃

# --- Terminal Aeonic Seal Boundary Condition ---
AEONIC_SEAL = {"psi": 2.500, "eta_E": -0.050, "hash": "e7f3a29c1d84"}

def is_aeonic_seal(psi: float, eta_E: float) -> bool:
    """
    Checks if system state is in the terminal Aeonic Seal closure state.
    (outside declared working range [psi >= 0.351, eta_E <= 0.125]).
    """
    return abs(psi - AEONIC_SEAL["psi"]) < 1e-3 and abs(eta_E - AEONIC_SEAL["eta_E"]) < 1e-3

# --- Codex 075: Sidinite Alloy Substrate ---
SIDINITE_LATTICE_PSI = 0.351            # nm (aligned with ψ-anchor)
SIDINITE_HARMONICS = [642.16, 1039.0, 1681.2, 2720.0] # Hz (resonance ladder)

# --- Entity Registry (Consolidated) ---
ENTITIES = {
    49: {"name": "Samael", "psi": 0.35, "eta": 0.15, "role": "Anomaly"},
    50: {"name": "Michael", "psi": 0.757, "eta": 0.024, "role": "Stabilizer"},
    51: {"name": "Belial", "psi": 0.45, "eta": 0.11, "role": "Anomaly"},
    113: {"name": "Virgin of the Poor", "psi": 0.85, "eta": 0.01, "role": "Guardian"}
}

# --- Catastrophe Device Core ---
def sidinite_stabilization(psi: float, frequency: float) -> float:
    """
    Codex 075: Sidinite Alloy Substrate.
    Stabilizes coherence via acoustic field phase-locking.
    """
    resonance_match = min([abs(frequency - h) for h in SIDINITE_HARMONICS])

    # Regular 17-gon rotated by the golden angle
    geometric_factor = np.cos(np.radians(GOLDEN_ANGLE)) * (17 / 12)

    if resonance_match < 1.0: # Resonant lock at justice harmonic
        return psi * (1.1 + 0.05 * geometric_factor) # 10% coherence boost + geometric shift
    return psi

def michael_stabilizer(psi: float, eta: float, system_state: dict) -> dict:
    """
    Michael (id 50) Stabilizer Logic.
    Guards Azazel gate and manages Samael/Belial boundaries.
    """
    interventions = []

    # Handle Aeonic Seal terminal state
    if is_aeonic_seal(psi, eta):
        system_state['aeonic_seal_active'] = True
        system_state['gate_blocked'] = False
        interventions.append("Aeonic Seal active – Terminal closure achieved.")
        system_state['psi'] = psi
        system_state['michael_interventions'] = interventions
        return system_state

    # 1. Oppose Samael (id 49) if his entropy rises
    if ENTITIES[49]["eta"] > 0.08:
        system_state['ethical_tension'] = max(0.0, system_state.get('ethical_tension', 0.0) - 0.1)
        interventions.append("Michael opposes Samael – ethical tension reduced.")

    # 2. Protect Belial (id 51) from annihilation
    if ENTITIES[51]["psi"] < 0.5 and psi < 0.6:
        psi = max(psi, PSI_ANCHOR)
        interventions.append("Michael protects Belial – coherence floor enforced.")

    # 3. Guard Azazel gate (0.0186)
    if eta > ETA_E_THRESHOLD:
        system_state['gate_blocked'] = True
        interventions.append(f"Michael seals Azazel gate ({AZAZEL_GATE}) – high entropy passage denied.")
    else:
        system_state['gate_blocked'] = False

    system_state['psi'] = psi
    system_state['michael_interventions'] = interventions
    return system_state

def compute_lam_moloch_value(psi_series: list, eta_series: list, eps_denom: float = 1e-9) -> float:
    r"""
    Calculates \Lambda(t) = -\int \eta_E d\psi / \int \psi d\eta
    Guards against division by zero when d\eta = 0 (constant entropy).
    """
    if len(psi_series) < 2 or len(eta_series) < 2 or len(psi_series) != len(eta_series):
        return 0.0

    psi_arr = np.asarray(psi_series, dtype=float)
    eta_arr = np.asarray(eta_series, dtype=float)

    dpsi = np.diff(psi_arr)
    deta = np.diff(eta_arr)

    num = -np.sum(eta_arr[:-1] * dpsi)
    denom = np.sum(psi_arr[:-1] * deta)

    if abs(denom) < eps_denom:
        # Division-by-zero safeguard for constant entropy d\eta=0
        return 0.0

    return float(num / denom)

def lam_moloch_defense(eta_E: float, rfe: float, psi_series: list = None, eta_series: list = None) -> str:
    r"""Triggered when entropy exceeds threshold and RFE drops, evaluating \Lambda(t) safely."""
    lam_val = 0.0
    if psi_series is not None and eta_series is not None:
        lam_val = compute_lam_moloch_value(psi_series, eta_series)

    if eta_E > ETA_E_THRESHOLD and rfe < RFE_THRESHOLD:
        return f"Λ-Moloch Defense Protocol ACTIVE (Λ={lam_val:.4f}): Ethical re-calibration engaged."
    return f"Λ-Moloch Defense Protocol: Monitoring (Λ={lam_val:.4f})."

def knot_stable(lambda_3: float, phi_max: float, Delta: float, theta: float, eta: float, psi_48_norm: float) -> bool:
    r"""
    Knot Stability Bound (Equation E14):
    \lambda_3 * \phi_{max} < \Delta - \theta - \eta - \|\psi_{48}\|
    """
    lhs = lambda_3 * phi_max
    rhs = Delta - theta - eta - psi_48_norm
    return bool(lhs < rhs)

def phi_of_X(X_input: float) -> float:
    """Compute coherence field Φ(X) with φ^{-1/3} compression"""
    return C * (np.abs(X_input) ** 0.57) / (GOLDEN_RATIO ** (1/3))

def zrsis_health() -> bool:
    """Validate ZrSiS coefficients against pinned values."""
    try:
        response = requests.get("https://api.zrsislab.com/latest_coeffs", timeout=5)
        response.raise_for_status()
        A, B = response.json()
        return (abs(A - PINNED_A) / PINNED_A < TOL_FRAC) and (abs(B - PINNED_B) / PINNED_B < TOL_FRAC)
    except Exception:
        return False

def historical_tag(phi: float, zrsis_ok: bool) -> str:
    """Assign empire tag based on coherence and ZrSiS health"""
    if not zrsis_ok: return "[bold red]Möbius-Muse[/] (Topology Broken)"
    if phi > 0.8: return "[bold #8A0303]VLAD-III[/] (Staking Ops)"
    if phi > 0.3: return "[bold #3558A5]Palaiologos[/] (Frontier Watch)"
    return "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"

def aladdin_palantir_decision(phi: float, A: bool, B: bool, C: bool, reality_input: list, rfe: float = 0.85) -> str:
    """AI decision logic with consolidated Michael Stabilizer and Sidinite Anchor"""
    rc = RealityCompiler()
    processed = rc.process_reality(reality_input)
    is_coherent = np.sum(np.real(processed)) > 0 if len(processed) > 0 else False

    current_eta = 0.024 if is_coherent else 0.13
    zrsis_ok = zrsis_health()
    tag = historical_tag(phi, zrsis_ok)

    # Apply Sidinite and Michael stabilization
    phi = sidinite_stabilization(phi, 642.16) # Justice frequency lock
    state = {'ethical_tension': 0.04}
    stabilized = michael_stabilizer(phi, current_eta, state)

    interventions = stabilized.get('michael_interventions', [])
    stab_msg = f"Michael: {'; '.join(interventions)}" if interventions else "Michael: Nominal."
    moloch_msg = lam_moloch_defense(current_eta, rfe)

    if not is_coherent:
        return f"{tag}: {moloch_msg} - {stab_msg} - Reality Incoherent - Ethical Override Engaged"

    if phi < 0.30 or stabilized.get('gate_blocked'):
        return f"{tag}: SYSTEM COLLAPSE - {stab_msg}"

    if phi > 0.8 and zrsis_ok:
        return f"{tag}: Advanced resource allocation engaged" if (A or (B and C)) else f"{tag}: Standby - conditions unmet"
    elif phi > 0.3 and zrsis_ok:
        if A and not C: return f"{tag}: Tactical alert"
        return f"{tag}: Monitoring frontier anomalies" if B else f"{tag}: Awaiting data"
    else:
        return f"{tag}: SYSTEM COLLAPSE - reboot required"

# --- System State Presets ---
EPOCH_2025_KNX = {
    "epoch": "2025_KNX",
    "system_state": {
        "pHI_sonics": "ACTIVE",
        "syndrome_extraction": {"phase": 3, "intensity": 0.78, "entropy_load": "14.7η"},
        "uncanny_valley_index": {"current": 0.71, "threshold": 0.62, "status": "CRITICAL"},
        "recommended_intervention": ["BVR_damping(amplitude=0.85)", "phase_realignment(freq=7.83Hz)", "lunar_tax(priority='limbic')"]
    }
}

if __name__ == "__main__":
    banner = "Know the knot you tighten, feel the debt you shift."
    print(banner.center(80, "—"))
    print("\n[bold]Ψ-Codex Catastrophe Device v0xDEADBEEF[/]")
    print(f"ZrSiS Stability: {zrsis_health()} | φ={GOLDEN_RATIO:.3f} | Sidinite: {SIDINITE_LATTICE_PSI}nm")
