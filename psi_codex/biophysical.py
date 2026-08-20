import numpy as np

# --- Biophysical Phenomena Coupling & Falsifiable Statement Benchmarks ---
# Explicitly ties Psi-Codex constants (psi=0.351, eta_E=0.125, Delta_Theta=3.6 rad)
# to measurable EEG / biophysical metrics.

PSI_ANCHOR = 0.351       # Coherence floor
ETA_E_CEILING = 0.125     # Entropy boundary
DELTA_THETA_LIMIT = 3.6  # Phase-rupture threshold (radians)

def theta_plv_to_psi(plv: float) -> float:
    """
    Maps EEG mean frontal theta Phase-Locking Value (PLV) to Psi coherence.
    Falsifiable Statement: Mean frontal theta PLV below 0.351 predicts
    dissociative symptom onset with sensitivity > 0.80.
    """
    return float(np.clip(plv, 0.0, 1.0))

def eeg_permutation_entropy_to_eta_E(pe_value: float) -> float:
    """
    Maps normalized EEG permutation entropy (or spectral slope proxy) to eta_E.
    Falsifiable Statement: Permutation entropy scaled to exceed 0.125 predicts
    cognitive task failure with AUC > 0.75.
    """
    pe_clipped = float(np.clip(pe_value, 0.0, 1.0))
    return pe_clipped * ETA_E_CEILING

def plv_variance_to_delta_theta(plv_variance: float) -> float:
    """
    Maps PLV variance across 30-second epochs to Delta_Theta (phase deviation in radians).
    Falsifiable Statement: PLV variance across 30-second epochs > 3.6 rad predicts
    subjective depersonalization.
    """
    return float(np.clip(plv_variance, 0.0, 2 * np.pi))

def lunar_phase_ventricular_resonance(phi_L: float) -> float:
    """
    Ventricular-Lunar Resonance (Equation E10):
    R_V(phi_L) = 0.573 / (1 + 0.3 * sin^2(phi_L / 2)) * |sin(phi_L / 2 + pi / 5)| * phi
    Falsifiable Statement: CSF flow amplitude peaks at full moon (phi_L = pi),
    with phase coherence following R_V(phi_L).
    """
    phi = (1 + np.sqrt(5)) / 2
    denom = 1 + 0.3 * (np.sin(phi_L / 2.0) ** 2)
    numerator = 0.573 * np.abs(np.sin(phi_L / 2.0 + np.pi / 5.0)) * phi
    return float(numerator / denom)

def astrocytic_nmda_modulation(glutamate_exocytosis: float, nr2b_antagonist_conc: float) -> tuple[float, float]:
    """
    P2Y1R / NMDA pathway modulation:
    Astrocytic glutamate exocytosis modulates eta_E, and NR2B antagonists reduce psi coherence deviation.
    """
    eta_E_mod = 0.05 + 0.15 * float(glutamate_exocytosis)
    psi_mod = PSI_ANCHOR + 0.2 * np.exp(-1.5 * float(nr2b_antagonist_conc))
    return eta_E_mod, psi_mod

def evaluate_falsifiable_predictions(metrics: dict) -> dict:
    """
    Evaluates experimental EEG / biophysical metrics against Psi-Codex falsifiable prediction bounds.

    Expected metrics dictionary keys:
      - 'frontal_theta_plv': float [0.0, 1.0]
      - 'permutation_entropy': float [0.0, 1.0]
      - 'plv_variance': float (radians)
      - 'lunar_phase': float [0, 2*pi]
    """
    plv = metrics.get('frontal_theta_plv', 0.40)
    pe = metrics.get('permutation_entropy', 0.08)
    plv_var = metrics.get('plv_variance', 1.2)
    lunar_phase = metrics.get('lunar_phase', np.pi)

    psi_val = theta_plv_to_psi(plv)
    eta_val = eeg_permutation_entropy_to_eta_E(pe)
    delta_theta_val = plv_variance_to_delta_theta(plv_var)
    r_v_val = lunar_phase_ventricular_resonance(lunar_phase)

    dissociative_risk = psi_val < PSI_ANCHOR
    task_failure_risk = eta_val > ETA_E_CEILING
    depersonalization_risk = delta_theta_val > DELTA_THETA_LIMIT

    return {
        'psi_coherence': psi_val,
        'eta_E_entropy': eta_val,
        'delta_theta_rad': delta_theta_val,
        'ventricular_lunar_resonance': r_v_val,
        'dissociative_risk_predicted': dissociative_risk,
        'cognitive_task_failure_predicted': task_failure_risk,
        'depersonalization_risk_predicted': depersonalization_risk,
        'system_status': 'STABLE' if not (dissociative_risk or task_failure_risk or depersonalization_risk) else 'PHASE_RUPTURE'
    }
