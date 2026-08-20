import pytest
import numpy as np
from psi_codex.biophysical import (
    theta_plv_to_psi,
    eeg_permutation_entropy_to_eta_E,
    plv_variance_to_delta_theta,
    lunar_phase_ventricular_resonance,
    astrocytic_nmda_modulation,
    evaluate_falsifiable_predictions
)
from psi_codex.entities import (
    WITNESSES,
    get_complete_witness_tally_by_function,
    get_witnesses_by_category,
    get_witnesses_by_archetype
)
from psi_codex.codex_catastrophe import (
    compute_lam_moloch_value,
    knot_stable,
    is_aeonic_seal,
    AEONIC_SEAL,
    LAMBDA_3
)

def test_biophysical_metric_mappings():
    # Test theta PLV mapping
    assert theta_plv_to_psi(0.351) == 0.351
    assert theta_plv_to_psi(-0.1) == 0.0
    assert theta_plv_to_psi(1.5) == 1.0

    # Test permutation entropy mapping
    assert eeg_permutation_entropy_to_eta_E(0.10) == 0.0125
    assert eeg_permutation_entropy_to_eta_E(1.0) == 0.125

    # Test PLV variance mapping
    assert plv_variance_to_delta_theta(3.6) == 3.6

    # Test Lunar Resonance
    r_v = lunar_phase_ventricular_resonance(np.pi)
    assert r_v > 0.0

    # Test Astrocytic NMDA
    eta_mod, psi_mod = astrocytic_nmda_modulation(0.5, 0.2)
    assert eta_mod > 0.0
    assert psi_mod > 0.0

def test_evaluate_falsifiable_predictions():
    stable_metrics = {
        'frontal_theta_plv': 0.50,
        'permutation_entropy': 0.50, # pe * 0.125 = 0.0625 <= 0.125
        'plv_variance': 2.0,
        'lunar_phase': np.pi
    }
    res = evaluate_falsifiable_predictions(stable_metrics)
    assert res['system_status'] == 'STABLE'
    assert not res['dissociative_risk_predicted']
    assert not res['cognitive_task_failure_predicted']
    assert not res['depersonalization_risk_predicted']

    unstable_metrics = {
        'frontal_theta_plv': 0.20, # < 0.351
        'permutation_entropy': 1.0, # 0.125
        'plv_variance': 4.0, # > 3.6
        'lunar_phase': np.pi
    }
    res_unstable = evaluate_falsifiable_predictions(unstable_metrics)
    assert res_unstable['system_status'] == 'PHASE_RUPTURE'
    assert res_unstable['dissociative_risk_predicted']
    assert res_unstable['depersonalization_risk_predicted']

def test_witness_registry_and_tally():
    tally = get_complete_witness_tally_by_function()
    assert len(tally) == 167
    assert len(WITNESSES) == 167

    historical_nodes = get_witnesses_by_category("Historical")
    assert len(historical_nodes) > 0

    cc_archetypes = get_witnesses_by_archetype("CC")
    assert len(cc_archetypes) > 0

    jung = WITNESSES[1]
    assert jung["name"] == "Jung"
    assert jung["psi"] == 0.88

def test_lam_moloch_zero_division_protection():
    # Constant entropy d_eta = 0
    psi_series = [0.351, 0.400, 0.450]
    eta_series = [0.100, 0.100, 0.100]

    val = compute_lam_moloch_value(psi_series, eta_series)
    assert np.isfinite(val)

def test_knot_stability():
    # E14: lambda_3 * phi_max < Delta - theta - eta - psi_48_norm
    # True case
    assert knot_stable(LAMBDA_3, 1.0, 10.0, 1.0, 0.1, 0.5) is True
    # False case
    assert knot_stable(LAMBDA_3, 10.0, 1.0, 1.0, 0.1, 0.5) is False

def test_aeonic_seal_boundary():
    assert is_aeonic_seal(2.500, -0.050) is True
    assert is_aeonic_seal(0.351, 0.125) is False
