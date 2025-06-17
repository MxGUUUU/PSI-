import pytest
import numpy as np
from pytest_mock import mocker # Explicit import for clarity, or rely on pytest's built-in if preferred
from psi_codex.codex_catastrophe import phi_of_X, aladdin_palantir_decision, historical_tag

# --- Unit Tests for codex_catastrophe.py ---

def test_phi_monotone():
    """Φ(X) should be non-decreasing with |X|"""
    xs = np.linspace(0.0, 3.0, 10)
    phis = [phi_of_X(x) for x in xs]
    # Check that each element is greater than or equal to the previous one
    assert all(phis[i] <= phis[i+1] for i in range(len(phis)-1)), \
        "Φ(X) is not non-decreasing with |X|"

def test_decision_branches(mocker): # mocker fixture is automatically provided by pytest-mock
    """Test the main decision branches of aladdin_palantir_decision, mocking zrsis_health"""

    # Test case 1: VLAD-III branch (High phi, ZrSiS OK)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=True)
    # phi_of_X(2.5) is > 0.8. Let's use a direct phi value for clarity in test.
    # phi_val_vlad = phi_of_X(2.5) # approx 0.85
    phi_val_vlad = 0.9
    assert "VLAD-III" in aladdin_palantir_decision(phi_val_vlad, True, False, False)
    assert "Advanced resource allocation engaged" in aladdin_palantir_decision(phi_val_vlad, True, False, False)
    assert "Standby - conditions unmet" in aladdin_palantir_decision(phi_val_vlad, False, False, False)


    # Test case 2: Palaiologos branch (Mid phi, ZrSiS OK)
    # phi_val_palaiologos = phi_of_X(1.0) # approx 0.61
    phi_val_palaiologos = 0.5
    assert "Palaiologos" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False) # A=True, C=False -> Tactical alert
    assert "Tactical alert" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)
    assert "Monitoring frontier anomalies" in aladdin_palantir_decision(phi_val_palaiologos, False, True, False) # B=True
    assert "Awaiting data" in aladdin_palantir_decision(phi_val_palaiologos, False, False, False) # Default for mid phi

    # Test case 3: Opium-Raj branch (Low phi, ZrSiS OK)
    # phi_val_opium = phi_of_X(0.1) # approx 0.28
    phi_val_opium = 0.2
    assert "Opium-Raj" in aladdin_palantir_decision(phi_val_opium, False, False, False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_opium, False, False, False)

    # Test case 4: Möbius-Muse branch (ZrSiS Fails, high phi example)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=False)
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_vlad, True, False, False) # phi > 0.8
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_vlad, True, False, False) # because historical_tag is Möbius-Muse

    # Test case 5: Möbius-Muse branch (ZrSiS Fails, mid phi example)
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False) # 0.3 < phi <= 0.8
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)

    # Test case 6: Möbius-Muse branch (ZrSiS Fails, low phi example)
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_opium, True, False, False) # phi <= 0.3
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_opium, True, False, False)

def test_historical_tag_logic():
    """Test the historical_tag function directly."""
    # ZrSiS OK cases
    assert historical_tag(0.9, True) == "[bold #8A0303]VLAD-III[/] (Staking Ops)"
    assert historical_tag(0.5, True) == "[bold #3558A5]Palaiologos[/] (Frontier Watch)"
    assert historical_tag(0.2, True) == "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"

    # ZrSiS Not OK cases
    assert historical_tag(0.9, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
    assert historical_tag(0.5, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
    assert historical_tag(0.2, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
