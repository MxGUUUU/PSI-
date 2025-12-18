import numpy as np
from psi_codex.reality_compiler import RealityCompiler, ethical_coherence, reality_compilation, ZETA_3

def test_ethical_coherence():
    """
    Tests the ethical_coherence function to ensure it returns a modified array.
    """
    spatial_pattern = np.random.rand(100)
    coherent_pattern = ethical_coherence(spatial_pattern)
    assert coherent_pattern is not None
    assert len(coherent_pattern) == len(spatial_pattern)
    # Check that the returned array is different from the input,
    # as the filter should have modified it.
    assert not np.allclose(spatial_pattern, coherent_pattern)

def test_reality_compilation():
    """
    Tests the reality_compilation function.
    """
    psi_field = np.array([0.5, 0.5])
    phi_val = 0.8
    reality_vector = reality_compilation(psi_field, phi_val)
    assert reality_vector is not None
    assert len(reality_vector) == len(psi_field)
    # Expected value: 1.0 * (0.5**0.014) * 0.5 * 0.8 = 0.396137...
    assert np.allclose(reality_vector, [0.396137, 0.396137], atol=1e-6)

def test_reality_compiler_justice_operator():
    """
    Tests the RealityCompiler's justice_operator with coherent and incoherent inputs.
    """
    compiler = RealityCompiler()

    # Coherent input (close to ZETA_3)
    coherent_input = np.random.rand(100) * 2 * ZETA_3
    coherent_input = coherent_input / np.mean(np.abs(coherent_input)) * ZETA_3
    processed_coherent = compiler.justice_operator(coherent_input)
    assert np.allclose(coherent_input, processed_coherent)

    # Incoherent input (far from ZETA_3)
    incoherent_input = np.random.rand(100) * 100
    processed_incoherent = compiler.justice_operator(incoherent_input)
    assert not np.allclose(incoherent_input, processed_incoherent)

def test_reality_compiler_process_reality():
    """
    Tests the full process_reality method of the RealityCompiler.
    """
    compiler = RealityCompiler()
    raw_input = np.random.rand(100)
    phi_val = 0.8
    processed_reality = compiler.process_reality(raw_input, phi_val)
    assert processed_reality is not None
    assert len(processed_reality) == len(raw_input)

def test_ethical_coherence_with_empty_input():
    """
    Tests the ethical_coherence function with empty input.
    """
    coherent_pattern = ethical_coherence([])
    assert isinstance(coherent_pattern, np.ndarray)
    assert len(coherent_pattern) == 0

def test_reality_compiler_with_empty_input():
    """
    Tests the RealityCompiler with empty input.
    """
    compiler = RealityCompiler()
    processed_reality = compiler.process_reality([], 0.8)
    assert isinstance(processed_reality, np.ndarray)
    assert len(processed_reality) == 0
