import pytest
import numpy as np
from pytest_mock import mocker # pytest-mock provides the mocker fixture
from psi_codex.codex_catastrophe import phi_of_X, aladdin_palantir_decision, historical_tag
from psi_codex.reality_compiler import ZETA_3

# --- Unit Tests for codex_catastrophe.py ---

def test_phi_monotone():
    """Φ(X) should be non-decreasing with |X|"""
    xs = np.linspace(0.0, 3.0, 10)
    phis = [phi_of_X(x) for x in xs]
    # Check that each element is greater than or equal to the previous one
    # Allow for very small floating point inaccuracies by checking phis[i] <= phis[i+1] + epsilon
    epsilon = 1e-9
    assert all(phis[i] <= phis[i+1] + epsilon for i in range(len(phis)-1)), \
        "Φ(X) is not non-decreasing with |X|"

def test_decision_branches(mocker): # mocker fixture is automatically provided by pytest-mock
    """Test the main decision branches of aladdin_palantir_decision, mocking zrsis_health"""

    # Create a reality input that is coherent (mean absolute value is close to ZETA_3)
    coherent_reality_input = np.random.rand(100) * 2 * ZETA_3
    coherent_reality_input = coherent_reality_input / np.mean(np.abs(coherent_reality_input)) * ZETA_3

    # Test case 1: VLAD-III branch (High phi, ZrSiS OK, Coherent)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=True)
    phi_val_vlad = 0.9
    assert "Advanced resource allocation engaged" in aladdin_palantir_decision(phi_val_vlad, True, False, False, coherent_reality_input)

    # Test case 2: Incoherent reality
    incoherent_reality_input = np.random.rand(100) * 100 # Far from ZETA_3
    assert "Ethical Override Engaged" in aladdin_palantir_decision(phi_val_vlad, True, False, False, incoherent_reality_input)

    # Test case 3: Palaiologos branch (Mid phi, ZrSiS OK, Coherent)
    phi_val_palaiologos = 0.5
    assert "Tactical alert" in aladdin_palantir_decision(phi_val_palaiologos, True, False, False, coherent_reality_input)

    # Test case 4: System Collapse (Low phi)
    phi_val_opium = 0.2
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_opium, False, False, False, coherent_reality_input)

    # Test case 5: System Collapse (ZrSiS Fails)
    mocker.patch('psi_codex.codex_catastrophe.zrsis_health', return_value=False)
    assert "SYSTEM COLLAPSE" in aladdin_palantir_decision(phi_val_vlad, True, False, False, coherent_reality_input)

def test_historical_tag_logic():
    """Test the historical_tag function directly for all branches."""
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
