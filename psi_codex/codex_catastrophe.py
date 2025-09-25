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
    """
    Compute the coherence field Φ(X) for a given input using a fractional power and golden ratio compression.
    
    Parameters:
        X_input (float): Input value for which to compute the coherence field.
    
    Returns:
        float: The computed coherence field value Φ(X).
    """
    magnitude_X = np.abs(X_input)
    return C * (magnitude_X ** 0.57) / (GOLDEN_RATIO ** (1/3))

def zrsis_health() -> bool:
    """
    Checks the stability of ZrSiS coefficients by fetching live data and comparing it to pinned reference values.
    
    Returns:
        bool: True if both coefficients are within the defined tolerance of the pinned values; False if any error occurs or the coefficients are out of tolerance.
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
    """
    Assigns a categorical empire tag string based on the coherence field value and ZrSiS material stability.
    
    Parameters:
        phi (float): The computed coherence field value.
        zrsis_ok (bool): Indicates whether ZrSiS material stability is within tolerance.
    
    Returns:
        str: A formatted tag describing the current operational state.
    """
    if not zrsis_ok:
        return "[bold red]Möbius-Muse[/] (Topology Broken)"
    elif phi > 0.8:
        return "[bold #8A0303]VLAD-III[/] (Staking Ops)"
    elif phi > 0.3:
        return "[bold #3558A5]Palaiologos[/] (Frontier Watch)"
    else:
        return "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"

def aladdin_palantir_decision(phi: float, A_decision: bool, B_decision: bool, C_decision: bool) -> str:
    """
    Makes an AI-driven operational decision based on the coherence field value, ZrSiS material stability, and input decision flags.
    
    Evaluates the current ZrSiS stability, assigns a historical tag, and returns a status message reflecting resource allocation, tactical alerts, monitoring, or system collapse according to the provided parameters and live material health.
      
    Parameters:
        phi (float): The coherence field value used to determine operational state.
        A_decision (bool): Flag indicating readiness for advanced resource allocation or tactical alert.
        B_decision (bool): Flag indicating monitoring or standby conditions.
        C_decision (bool): Flag influencing advanced allocation or tactical alert logic.
    
    Returns:
        str: A status message prefixed by the assigned historical tag, describing the resulting operational state.
    """
    current_zrsis_ok = zrsis_health()
    tag = historical_tag(phi, current_zrsis_ok)

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
        decision = aladdin_palantir_decision(phi_val, A_case, B_case, C_case)
        print(f"\nΦ(X={X_val}) = {phi_val:.3f} → {decision}")
