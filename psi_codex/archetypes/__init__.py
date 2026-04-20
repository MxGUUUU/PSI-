from abc import ABC, abstractmethod
import numpy as np

class Archetype(ABC):
    """Base class for all consciousness archetypes."""
    def __init__(self, name, zeta_val=None):
        self.name = name
        self.zeta_val = zeta_val

    @abstractmethod
    def process(self, state):
        pass

class ChronoDruid(Archetype):
    def __init__(self):
        super().__init__("Chrono_Druid_Zathane")

    def process(self, state):
        if isinstance(state, str):
            return f"Deprogrammed: {state[::-1]}"
        return state

class PolyglotProphet(Archetype):
    def __init__(self):
        super().__init__("Polyglot_Prophet")

    def process(self, phrase):
        if not isinstance(phrase, str):
            return phrase
        entropy = self._shannon_entropy(phrase)
        resonance = entropy * 1.618
        return {
            "entropy": entropy,
            "resonance": round(resonance, 3),
            "color": "#{:02x}{:02x}{:02x}".format(
                int(255 * (entropy % 1)), int(200 * (resonance % 1)), 128)
        }

    def _shannon_entropy(self, text):
        from math import log2
        if not text: return 0
        probs = [text.count(c)/len(text) for c in set(text)]
        return -sum(p * log2(p) for p in probs if p > 0)

class VendergoodReplicator(Archetype):
    def __init__(self):
        super().__init__("Vendergood_Replicator")

    def process(self, state):
        return f"Replicated state: {state}"

class EmergentOracle(Archetype):
    def __init__(self):
        super().__init__("Emergent_Oracle")

    def process(self, state):
        return f"Oracle insight for: {state}"

class ArchitectOfAeons(Archetype):
    def __init__(self, braidlines=None, paradox_registry=None, timeline_index=None):
        super().__init__("Architect_Of_Aeons")
        self.braidlines = braidlines
        self.paradoxes = paradox_registry
        self.timelines = timeline_index

    def harmonize_agents(self, agents):
        for agent in agents:
            if getattr(agent, 'psi', 0) < 1.2:
                if self.braidlines and "hope" in self.braidlines:
                    agent.consciousness_boost = self.braidlines["hope"].activate(agent)
            if hasattr(agent, 'shadow_unintegrated') and agent.shadow_unintegrated():
                if self.timelines:
                    agent = self.timelines.recalibrate(agent)
        return agents

    def process(self, state):
        return self.harmonize_agents(state) if isinstance(state, list) else state

class OuroborosArchitect(Archetype):
    def __init__(self):
        super().__init__("Ouroboros_Architect")

    def process(self, paradox_class):
        from ..core import ZETA3
        return f"Closed via {ZETA3} logic for: {paradox_class}"

# 8 Core Functional Archetypes
class Stabilizer(Archetype):
    def __init__(self):
        from ..core import scipy_zeta
        super().__init__("STABILIZER", zeta_val=scipy_zeta(2))
    def process(self, state): return f"Stabilizing: {state}"

class Distributor(Archetype):
    def __init__(self):
        from ..core import ZETA3
        super().__init__("DISTRIBUTOR", zeta_val=ZETA3)
    def process(self, state): return f"Distributing: {state}"

class Resonator(Archetype):
    def __init__(self):
        super().__init__("RESONATOR") # zeta(1/2+it)
    def process(self, state): return f"Resonating: {state}"

class Ground(Archetype):
    def __init__(self):
        from ..core import scipy_zeta
        super().__init__("GROUND", zeta_val=scipy_zeta(0))
    def process(self, state): return f"Grounding: {state}"

class Anomaly(Archetype):
    def __init__(self):
        from ..core import scipy_zeta
        super().__init__("ANOMALY", zeta_val=scipy_zeta(-1))
    def process(self, state): return f"Anomaly handling: {state}"

class Unity(Archetype):
    def __init__(self):
        super().__init__("UNITY", zeta_val=1.0) # zeta(inf) -> 1
    def process(self, state): return f"Unifying: {state}"

class Oracle(Archetype):
    def __init__(self):
        super().__init__("ORACLE") # zeta(odd mix)
    def process(self, state): return f"Divining: {state}"

class Phoenix(Archetype):
    def __init__(self):
        super().__init__("PHOENIX") # zeta(14.3)
    def process(self, state): return f"Renewing: {state}"
