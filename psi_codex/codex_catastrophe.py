import math
import numpy as np
import requests
from rich import print

# --- Ψ-Codex Constants ---
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ
C = 0.0573 * GOLDEN_RATIO               # Cruel-entropy thesis constant (0.057)
PINNED_A, PINNED_B = 0.348, 0.651       # ZrSiS nodal-line coefficients (defined internally)
TOL_FRAC = 0.05                         # 5% tolerance for material stability (defined internally)

# --- Catastrophe Device Core ---
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
        # In a real scenario, this URL would point to a live API
        # For now, this will likely fail unless a mock server is running
        # or this part is mocked during tests.
        response = requests.get("https://api.zrsislab.com/latest_coeffs", timeout=5)
        response.raise_for_status() # Raise an exception for HTTP errors
        A, B = response.json()  # Expecting a JSON response like [0.348, 0.651]

        ok_A = abs(A - PINNED_A) / PINNED_A < TOL_FRAC
        ok_B = abs(B - PINNED_B) / PINNED_B < TOL_FRAC
        return ok_A and ok_B
    except requests.exceptions.RequestException: # Catch network errors, timeout, bad status
        # print("[bold yellow]ZrSiS health check: Network/API error. Assuming unstable.[/]") # Optional: for debugging
        return False
    except Exception: # Catch other errors like JSON parsing, etc.
        # print("[bold yellow]ZrSiS health check: Error processing data. Assuming unstable.[/]") # Optional: for debugging
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

def aladdin_palantir_decision(phi: float, A_decision: bool, B_decision: bool, C_decision: bool) -> str: # Renamed A,B,C to avoid conflict
    """AI decision logic with ZrSiS stability enforcement"""
    # This zrsis_health call will be real or mocked by tests
    current_zrsis_ok = zrsis_health()
    tag = historical_tag(phi, current_zrsis_ok)

    if phi > 0.8 and current_zrsis_ok:
        if A_decision or (B_decision and C_decision):
            return f"{tag}: Advanced resource allocation engaged"
        return f"{tag}: Standby - conditions unmet"
    elif phi > 0.3 and current_zrsis_ok: # Ensure zrsis_ok is also checked here
        if A_decision and not C_decision:
            return f"{tag}: Tactical alert - threat pattern detected"
        elif B_decision:
            return f"{tag}: Monitoring frontier anomalies"
        return f"{tag}: Awaiting data"
    else: # This covers low phi OR not zrsis_ok
        return f"{tag}: SYSTEM COLLAPSE - reboot required"


if __name__ == "__main__":
    print("\n[bold]Ψ-Codex Catastrophe Device v0xDEADBEEF[/]")
    # The zrsis_health() call here will attempt a live request or fail gracefully.
    # For local testing without a live API, it will likely print "ZrSiS Stability: False"
    # unless the API endpoint happens to be up and returns expected data, or this is mocked.
    print(f"ZrSiS Stability (live check): {zrsis_health()} | φ={GOLDEN_RATIO:.3f}")

    test_cases = [
        (2.5, True, False, True),   # Vlad-III scenario
        (1.0, True, True, False),    # Palaiologos scenario
        (0.1, False, False, False),  # Opium-Raj scenario
        (3.0, False, False, False)   # Example where zrsis_health might be False
    ]

    print("\n--- Running Test Cases (assuming zrsis_health might be mocked or fail gracefully) ---")
    for X_val, A_case, B_case, C_case in test_cases:
        phi_val = phi_of_X(X_val)
        # For __main__ execution, aladdin_palantir_decision will call zrsis_health() itself.
        # If we wanted to test historical_tag independently of a live zrsis_health call for __main__,
        # we might pass a zrsis_ok status to it, e.g., historical_tag(phi_val, True) for some cases.
        # However, the current structure calls zrsis_health() inside aladdin_palantir_decision.
        decision = aladdin_palantir_decision(phi_val, A_case, B_case, C_case)
        print(f"\nΦ(X={X_val}) = {phi_val:.3f} → {decision}")
