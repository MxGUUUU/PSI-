import pytest
import numpy as np
from pytest_mock import mocker # pytest-mock provides the mocker fixture
from psi_codex.codex_catastrophe import phi_of_X, aladdin_palantir_decision, historical_tag

# --- Unit Tests for codex_catastrophe.py ---

def test_phi_monotone():
    """
    Test that the phi_of_X function is non-decreasing with respect to the absolute value of its input.
    
    Verifies that for a range of values from 0.0 to 3.0, the output of phi_of_X does not decrease, allowing for minor floating-point inaccuracies.
    """
    xs = np.linspace(0.0, 3.0, 10)
    phis = [phi_of_X(x) for x in xs]
    # Check that each element is greater than or equal to the previous one
    # Allow for very small floating point inaccuracies by checking phis[i] <= phis[i+1] + epsilon
    epsilon = 1e-9
    assert all(phis[i] <= phis[i+1] + epsilon for i in range(len(phis)-1)), \
        "Φ(X) is not non-decreasing with |X|"

def test_decision_branches(mocker): # mocker fixture is automatically provided by pytest-mock
    """
    Test all major decision branches of the aladdin_palantir_decision function by simulating different system health states and phi values.
    
    This test covers the VLAD-III, Palaiologos, Opium-Raj, and Möbius-Muse branches by mocking the zrsis_health dependency and varying decision flags. It asserts that the returned decision strings contain the expected keywords and status messages for each scenario.
    """

    # Test case 1: VLAD-III branch (High phi, ZrSiS OK)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=True)
    phi_val_vlad = 0.9
    # A_decision=True, B_decision=False, C_decision=False
    assert "VLAD-III" in aladdin_palantir_decision(phi_val_vlad, True, False, False)
    assert "Advanced resource allocation engaged" in aladdin_palantir_decision(phi_val_vlad, True, False, False)
    # A_decision=False, B_decision=True, C_decision=True
    assert "Advanced resource allocation engaged" in aladdin_palantir_decision(phi_val_vlad, False, True, True)
    # A_decision=False, B_decision=False, C_decision=False
    assert "Standby - conditions unmet" in aladdin_palantir_decision(phi_val_vlad, False, False, False)

    # Test case 2: Palaiologos branch (Mid phi, ZrSiS OK)
    phi_val_palaiologos = 0.5
    # A_decision=True, B_decision=False, C_decision=False
    assert "Palaiologos" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)
    assert "Tactical alert" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)
    # A_decision=False, B_decision=True, C_decision=False
    assert "Monitoring frontier anomalies" in aladdin_palantir_decision(phi_val_palaiologos, False, True, False)
    # A_decision=False, B_decision=False, C_decision=False
    assert "Awaiting data" in aladdin_palantir_decision(phi_val_palaiologos, False, False, False)

    # Test case 3: Opium-Raj branch (Low phi, ZrSiS OK)
    phi_val_opium = 0.2
    assert "Opium-Raj" in aladdin_palantir_decision(phi_val_opium, False, False, False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_opium, False, False, False)

    # Test case 4: Möbius-Muse branch (ZrSiS Fails, high phi example)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=False)
    # Decision for Möbius-Muse is always "SYSTEM COLLAPSE" if historical_tag indicates it.
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_vlad, True, False, False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_vlad, True, False, False)

    # Test case 5: Möbius-Muse branch (ZrSiS Fails, mid phi example)
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False)

    # Test case 6: Möbius-Muse branch (ZrSiS Fails, low phi example)
    assert "Möbius-Muse" in aladdin_palantir_decision(phi_val_opium, True, False, False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_opium, True, False, False)

def test_historical_tag_logic():
    """
    Test that the historical_tag function returns the correct tag string for all combinations of phi values and ZrSiS health status.
    
    Verifies that the tag output matches the expected label for each operational branch (VLAD-III, Palaiologos, Opium-Raj, Möbius-Muse) across boundary and typical phi values, and for both healthy and unhealthy system states.
    """
    # ZrSiS OK cases
    assert historical_tag(0.9, True) == "[bold #8A0303]VLAD-III[/] (Staking Ops)"
    assert historical_tag(0.81, True) == "[bold #8A0303]VLAD-III[/] (Staking Ops)" # Boundary for > 0.8

    assert historical_tag(0.8, True) == "[bold #3558A5]Palaiologos[/] (Frontier Watch)" # Boundary for <= 0.8
    assert historical_tag(0.5, True) == "[bold #3558A5]Palaiologos[/] (Frontier Watch)"
    assert historical_tag(0.31, True) == "[bold #3558A5]Palaiologos[/] (Frontier Watch)" # Boundary for > 0.3

    assert historical_tag(0.3, True) == "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"  # Boundary for <= 0.3
    assert historical_tag(0.2, True) == "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"
    assert historical_tag(0.0, True) == "[bold #7A6F45]Opium-Raj[/] (Entropy Drift)"

    # ZrSiS Not OK cases (phi value shouldn't matter for the tag itself, only for the decision logic that uses the tag)
    assert historical_tag(0.9, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
    assert historical_tag(0.5, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
    assert historical_tag(0.2, False) == "[bold red]Möbius-Muse[/] (Topology Broken)"
