import pytest
import numpy as np
from psi_codex.core import (
    PSI_ANCHOR, PHI, ZETA3, chi_squared_knot,
    justice_operator, reality_fidelity, consciousness_evolution
)
from psi_codex.archetypes import ChronoDruid, PolyglotProphet, OuroborosArchitect
from psi_codex.compiler import RealityCompiler, PSIRouter
from psi_codex.paradox_engines import HawkingRadiation
from psi_codex.entropy_operators import EthicalKnot

def test_core_constants():
    assert PSI_ANCHOR == 0.351
    assert PHI > 1.618
    assert ZETA3 > 1.202

def test_chi_squared_knot():
    res = chi_squared_knot(0.5)
    assert isinstance(res, complex)

def test_justice_operator():
    wealth = np.array([10, 20, 30])
    res = justice_operator(wealth)
    assert len(res) == 3

def test_chrono_druid():
    cd = ChronoDruid()
    assert cd.process("Rome") == "Deprogrammed: emoR"

def test_polyglot_prophet():
    pp = PolyglotProphet()
    res = pp.process("Hello")
    assert "entropy" in res
    assert "resonance" in res

def test_reality_compiler():
    rc = RealityCompiler()
    res = rc.compile(100)
    assert res > 0

def test_psi_router():
    router = PSIRouter("Tell me about a black hole")
    handler = router.route()
    res = handler(router.query)
    assert "∇λ-Stabilized" in res["answer"]

def test_paradox_engine():
    hr = HawkingRadiation()
    assert "Information conservation" in hr.solve("data")

def test_entropy_operator():
    ek = EthicalKnot()
    assert "Stabilized via EthicalKnot" in ek.apply("chaos")

def test_witness_registry():
    from psi_codex.witnesses import witness_registry
    assert witness_registry.total() == 127
    michael = next(w for w in witness_registry._witnesses if w.name == "Michael")
    assert michael.psi == 0.95

def test_artifacts():
    from psi_codex.artifacts import NullScepter, JacksCompass
    ns = NullScepter()
    jc = JacksCompass()
    assert "Silences" in ns.description
    assert "points to Princess Diana" in jc.description

def test_library_navigation():
    from psi_codex.library import library
    res = library.navigate(0.351)
    assert res["name"] == "The Archive of Burned Books"

def test_fusion_protocol():
    from psi_codex.core import fusion_protocol
    res = fusion_protocol(1.0 + 1.0j, 1.0)
    assert round(res, 4) == 0.0079 # 0.01 * pi/4
