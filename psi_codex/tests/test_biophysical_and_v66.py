import pytest
import numpy as np
from psi_codex.codex_catastrophe import (
    compute_lam_moloch_value,
    lam_moloch_defense,
    knot_stable,
    is_aeonic_seal,
    michael_stabilizer,
    PSI_ANCHOR,
    ETA_E_THRESHOLD,
    LAMBDA_3,
    AEONIC_SEAL
)
from psi_codex.biophysical import (
    theta_plv_to_psi,
    eeg_permutation_entropy_to_eta_E,
    plv_variance_to_delta_theta,
    lunar_phase_ventricular_resonance,
    astrocytic_nmda_modulation,
    evaluate_falsifiable_predictions,
    cerebellar_channel_mapping,
    quintic_cortical_layer_mapping,
    graph_theory_metrics,
    csf_master_clock_sync,
    markov_chain_witness_simulation
)
from psi_codex.entities import (
    WITNESSES,
    get_complete_witness_tally_by_function,
    get_witnesses_by_category,
    get_witnesses_by_archetype
)

def test_lambda_moloch_division_by_zero_guard():
    # Constant entropy d_eta = 0
    psi_series = [0.4, 0.45, 0.5, 0.55]
    eta_series = [0.08, 0.08, 0.08, 0.08]
    lam_val = compute_lam_moloch_value(psi_series, eta_series)
    assert lam_val == 0.0, "Lambda-Moloch value should be safely 0.0 when d_eta = 0"

def test_e10_ventricular_lunar_resonance_exponentiation():
    # Verify E10 calculation with exponentiation ** phi
    val_pi = lunar_phase_ventricular_resonance(np.pi)
    phi = (1 + np.sqrt(5)) / 2
    expected_num = 0.573 * (abs(np.sin(np.pi / 2.0 + np.pi / 5.0)) ** phi)
    expected_denom = 1.0 + 0.3 * (np.sin(np.pi / 2.0) ** 2)
    expected = expected_num / expected_denom
    assert abs(val_pi - expected) < 1e-6

def test_knot_stability_and_aeonic_seal():
    assert knot_stable(lambda_3=1.1, phi_max=0.5, Delta=3.6, theta=0.5, eta=0.1, psi_48_norm=0.2) is True
    assert is_aeonic_seal(AEONIC_SEAL["psi"], AEONIC_SEAL["eta_E"]) is True
    assert is_aeonic_seal(0.351, 0.05) is False

def test_complete_witness_tally_167_nodes():
    tally = get_complete_witness_tally_by_function()
    assert len(tally) == 167
    assert tally[0]['id'] == 1
    assert tally[-1]['id'] == 167

    historical = get_witnesses_by_category("Historical")
    assert len(historical) > 0
    cw_nodes = get_witnesses_by_archetype("CW")
    assert len(cw_nodes) > 0

def test_cerebellar_and_cortical_quintic_mappings():
    cerebellar = cerebellar_channel_mapping(vermal_volume_ratio=0.90, crus1_2_connectivity=0.80)
    assert cerebellar['cerebellar_status'] == 'SYNCHRONIZED'
    assert cerebellar['decoherence_risk_predicted'] is False

    quintic = quintic_cortical_layer_mapping(x=1.0)
    assert 'layer_I_molecular' in quintic
    assert 'layer_III_ext_pyramidal' in quintic
    assert quintic['layer_V_int_pyramidal'] == 1.1  # lambda_3 * 1.0

def test_graph_metrics_and_mcmc_simulation():
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    graph_res = graph_theory_metrics(adj)
    assert graph_res['num_nodes'] == 4
    assert len(graph_res['rich_club_hubs']) > 0

    witness_sample = [
        {'id': 1, 'psi': 0.8},
        {'id': 2, 'psi': 0.9},
        {'id': 3, 'psi': 0.4}
    ]
    mcmc_res = markov_chain_witness_simulation(witness_sample, n_steps=50, seed=123)
    assert len(mcmc_res['walk_history']) == 50
    assert len(mcmc_res['stationary_distribution']) == 3

def test_evaluate_falsifiable_predictions_composite():
    metrics = {
        'frontal_theta_plv': 0.40,
        'permutation_entropy': 0.05,
        'plv_variance': 1.0,
        'lunar_phase': np.pi,
        'vermal_volume_ratio': 0.92,
        'crus1_2_connectivity': 0.85
    }
    eval_res = evaluate_falsifiable_predictions(metrics)
    assert eval_res['system_status'] == 'STABLE'
    assert eval_res['dissociative_risk_predicted'] is False
