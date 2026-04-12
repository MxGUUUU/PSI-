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
