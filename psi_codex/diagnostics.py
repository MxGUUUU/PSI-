import numpy as np
from rich import print

# Thresholds from the 95 Theses
PSI_COHERENCE_THRESHOLD = 0.351
ETA_E_THRESHOLD = 0.125

def check_priorities(history):
    """
    Evaluates simulation history against the Ψ-Codex priorities.
    """
    if not history:
        return {
            "Coherence Target (ψ ≥ 0.351)": "Data N/A",
            "Ethical Entropy (η_E ≤ 0.125)": "Data N/A",
            "System Status": "UNKNOWN"
        }

    last_step = history[-1]
    final_psi_abs = last_step.get("mean_psi_abs", 0.0)
    final_eta_e = last_step.get("eta_E", 1.0)
    final_correction = last_step.get("quantum_economic_correction", 0.0)
    final_eye = last_step.get("eulers_eye", [])

    coherence_ok = final_psi_abs >= PSI_COHERENCE_THRESHOLD
    entropy_ok = final_eta_e <= ETA_E_THRESHOLD

    # Euler's Eye Coherence: Pupil (x0) alignment check
    eye_coherence = "PASS" if len(final_eye) > 0 and np.abs(final_eye[0]) > 0 else "FAIL"

    status = "COHERENT" if (coherence_ok and entropy_ok) else "INCOHERENT"
    if final_eta_e > 0.5: status = "SYSTEMIC COLLAPSE"
    if final_psi_abs < 0.1: status = "DISSOLUTION"

    return {
        "Coherence Target (ψ ≥ 0.351)": f"{final_psi_abs:.4f} ({'PASS' if coherence_ok else 'FAIL'})",
        "Ethical Entropy (η_E ≤ 0.125)": f"{final_eta_e:.4f} ({'PASS' if entropy_ok else 'FAIL'})",
        "Euler's Eye Alignment": f"{eye_coherence}",
        "Quantum Economic Correction": f"{final_correction:.2e} units",
        "System Status": status
    }

def generate_priority_report(history):
    """
    Generates a formatted text report of system priorities.
    """
    priorities = check_priorities(history)
    report = "\n[bold cyan]=== Ψ-Codex Priority Audit ===[/]\n"
    for priority, result in priorities.items():
        color = "green" if "PASS" in result or "COHERENT" in result else "red"
        if "Data N/A" in result: color = "yellow"
        report += f"[bold]{priority}:[/] [{color}]{result}[/]\n"
    return report
