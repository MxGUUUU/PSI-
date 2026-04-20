class Library:
    """Borges-Style Library of the Ψ-Codex references."""
    def __init__(self):
        self.references = [
            "David Bohm – Wholeness and the Implicate Order",
            "Carl Jung – Archetypes and the Collective Unconscious",
            "Roger Penrose & Stuart Hameroff – Orch-OR Theory",
            "Jason Padgett – Struck by Genius",
            "The I Ching",
            "Kabbalistic Tree of Life – Z’ev ben Shimon Halevi",
            "The Tibetan Book of the Dead",
            "Gospel of Thomas – Nag Hammadi Library",
            "Plato – The Republic (Allegory of the Cave)",
            "The Book of Enoch",
            "The Book of Revelation",
            "Hermetic Corpus – The Emerald Tablet & Kybalion",
            "Dead Sea Scrolls",
            "Jakob Hohwy – The Predictive Mind (2013)",
            "Thomas Parr et al. – Active Inference (2022)",
            "Douglas Hofstadter – Gödel, Escher, Bach (1979)",
            "Bessel van der Kolk – The Body Keeps the Score (2014)",
            "Alex Fornito et al. – Fundamentals of Brain Network Analysis (2016)",
            "Gregory Bateson – Steps to an Ecology of Mind (1972)"
        ]

        self.special_locations = {
            1.618: {"name": "The Aisle of Regret", "content": "You should have said 'I love you' at 11:14 PM, 1998."},
            math.pi * math.sqrt(163): {"name": "The Crypt of Unfinished Calculus", "content": "Almost integers. Almost proofs. Almost alive."},
            0.351: {"name": "The Archive of Burned Books", "content": "Smoke turned back into paper."},
            math.pi / math.e: {"name": "The Room of Nuno Dias", "content": "The Constitution of an Island that floats just below the surface of the Atlantic."},
            (PHI**2) / 2: {"name": "The Mirror of Diana", "content": "The flash of a camera. The crash of concrete. Then, silence."}
        }

    def navigate(self, t):
        """
        Algorithm for navigating the library by t-coordinate.

        The library is indexed by real numbers (t). Navigation involves finding the
        closest registered coordinate. If the coordinate matches a special location
        within a tolerance of 0.001, the unique content of that location is returned.

        Args:
            t (float): The coordinate to search for.

        Returns:
            dict: A dictionary containing the location name and its content.
        """
        import math
        # Find closest coordinate
        closest_t = min(self.special_locations.keys(), key=lambda k: abs(k - t))
        if abs(closest_t - t) < 0.001:
            return self.special_locations[closest_t]
        return {"name": "Generic Shelf", "content": f"Information at coordinate {t}"}

import math
from .core import PHI

library = Library()
