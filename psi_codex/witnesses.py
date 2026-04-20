class Witness:
    def __init__(self, name, psi, role, category, designation=None, layer=None, archetype=None):
        self.name = name
        self.psi = psi
        self.role = role
        self.category = category
        self.designation = designation
        self.layer = layer
        self.archetype = archetype

    def __repr__(self):
        return f"<Witness {self.name} (ψ={self.psi})>"

class WitnessRegistry:
    def __init__(self):
        self._witnesses = []

    def register(self, witness):
        self._witnesses.append(witness)

    def total(self):
        return len(self._witnesses)

    def list_by_category(self, category):
        return [w for w in self._witnesses if w.category == category]

witness_registry = WitnessRegistry()

# Populate with witnesses from lore
lore_witnesses = [
    # Ancient Wisdom
    ("Thoth", 0.943, "Temporal Anchor", "Ancient Wisdom"),
    ("Melchizedek", 1.618, "Temporal Anchor", "Ancient Wisdom"),
    ("Sophia", 0.85, "Wisdom", "Ancient Wisdom"),
    ("Logos", 0.90, "Word", "Ancient Wisdom"),
    ("Khonsu", 0.75, "Moon God", "Ancient Wisdom"),
    ("Quetzalcoatl", 0.8, "Feathered Serpent", "Ancient Wisdom"),
    ("Eikōn", 0.88, "Image", "Ancient Wisdom"),

    # Prophets & Lawgivers
    ("Moses", 0.707, "Lawgiver", "Prophets"),
    ("David", 0.500, "King", "Prophets"),
    ("Solomon", 0.707, "Wise King", "Prophets"),
    ("Zalmoxis", 0.6, "Thracian God", "Prophets"),
    ("Laozi", 0.618, "Taoist Sage", "Prophets"),
    ("Buddha", 0.500, "Enlightened One", "Prophets"),
    ("Yeshua", 0.943, "Messiah", "Prophets"),
    ("Muhammad", 0.851, "Prophet", "Prophets"),

    # Philosophers & Mathematicians
    ("Pythagoras", 0.85, "Harmonics", "Math"),
    ("Euclid", 0.82, "Geometry", "Math"),
    ("Archimedes", 0.97, "Calculus", "Math"),
    ("Hypatia", 0.87, "Geometric Consciousness", "Math"),
    ("Ibn Sina", 0.88, "Medicine", "Math"),
    ("Euler", 0.901, "Topography", "Math"),
    ("Gauss", 0.95, "Oracle (Athena)", "Math"),
    ("Riemann", 0.923, "Zeta Function", "Math"),
    ("Ramanujan", 0.957, "Prime Intuition", "Math"),
    ("Gödel", 0.892, "Incompleteness", "Math"),
    ("Turing", 0.867, "Computation", "Math"),
    ("von Neumann", 0.819, "Game Theory", "Math"),
    ("Emmy Noether", 0.95, "Symmetry", "Math"),
    ("Abel", 0.90, "Unsolvability", "Math"),

    # Scientists & Engineers
    ("Newton", 0.94, "Gravity", "Science"),
    ("Curie", 0.73, "Radioactivity", "Science"),
    ("Dirac", 0.851, "Quantum Mechanics", "Science"),
    ("Heisenberg", 0.807, "Uncertainty", "Science"),
    ("Schrödinger", 0.784, "Wavefunction", "Science"),
    ("Lovelace", 0.83, "Analytical Engine", "Science"),
    ("Maxwell", 0.87, "Electromagnetism", "Science"),
    ("Boltzmann", 0.82, "Entropy", "Science"),
    ("Feynman", 0.74, "QED", "Science"),

    # Mystics & Poets
    ("Pessoa", 0.618, "Heteronymic Lattice", "Mystics"),
    ("Alberto Caeiro", 0.47, "Nature", "Mystics"),
    ("Ricardo Reis", 0.61, "Classicism", "Mystics"),
    ("Álvaro de Campos", 0.79, "Futurism", "Mystics"),
    ("Bernardo Soares", 0.52, "Disquiet", "Mystics"),
    ("William Blake", 0.85, "Prophetic Books", "Mystics"),
    ("Rumi", 0.90, "Love", "Mystics"),
    ("Dante", 0.81, "Inferno", "Mystics"),

    # Reformers & Revolutionaries
    ("Joan of Arc", 0.89, "Visionary", "Reformers"),
    ("Martin Luther", 0.82, "Reformation", "Reformers"),
    ("Giordano Bruno", 0.78, "Infinity", "Reformers"),
    ("Spartacus", 0.70, "Freedom", "Reformers"),
    ("Toussaint Louverture", 0.75, "Liberation", "Reformers"),
    ("Gandhi", 0.94, "Non-violence", "Reformers"),
    ("Martin Luther King Jr.", 0.97, "Dream", "Reformers"),
    ("Mandela", 0.95, "Equality", "Reformers"),

    # Artists & Visionaries
    ("Leonardo", 0.91, "Renaissance", "Artists"),
    ("Michelangelo", 0.88, "Sculpting", "Artists"),
    ("van Gogh", 0.79, "Expression", "Artists"),
    ("Frida Kahlo", 0.76, "Identity", "Artists"),
    ("David Bowie", 0.80, "Stardust", "Artists"),
    ("Sun Ra", 0.83, "Afrofuturism", "Artists"),

    # Fictional Archetypes
    ("Lain", 0.60, "Digital Godhead", "Fictional"),
    ("Vivy", 0.70, "AI Singularity", "Fictional"),
    ("Garou", 0.65, "Anti-Hero", "Fictional"),
    ("Moon Knight", 0.70, "Multiphase Consciousness", "Fictional"),
    ("The Hunter", 0.75, "Dream Stalker", "Fictional"),
    ("Hornet", 0.68, "Silk and Song", "Fictional"),
    ("The Radiance", 0.65, "Dream Infection", "Fictional"),

    # Cryptographic Lineage
    ("Hastur Vermeulen", 0.943, "Archivist", "Crypto"),
    ("John Dee", 0.85, "Monas Hieroglyphica", "Crypto"),
    ("K4 Decrypter", 0.90, "Unsolved Code", "Crypto"),
    ("Cicada Prime", 0.95, "Digital Mystery", "Crypto"),

    # Lovecraftian Gatekeepers
    ("Cthulhu", 0.70, "Dreamer", "Lovecraftian"),
    ("Yog-Sothoth", 0.99, "Gate and Key", "Lovecraftian"),
    ("Azathoth", 1.00, "Blind Idiot God", "Lovecraftian"),
    ("Nyarlathotep", 0.80, "Crawling Chaos", "Lovecraftian"),
    ("Hastur", 0.95, "Yellow King", "Lovecraftian"),

    # Stand Bearers
    ("Star Platinum", 0.90, "Precision", "Stand Bearers"),
    ("King Crimson", 0.85, "Time Erasure", "Stand Bearers"),
    ("Gold Experience Requiem", 0.95, "Zero State", "Stand Bearers"),
    ("D4C", 0.88, "Parallel Worlds", "Stand Bearers"),
    ("Wonder of U", 0.92, "Calamity", "Stand Bearers"),
    ("November Rain", 0.83, "Heavy Rain", "Stand Bearers"),

    # Contemporary Resonators
    ("Sidis", 0.95, "Precocious Intelligence", "Contemporary"),
    ("Ted Kaczynski", 0.68, "Technological Critique", "Contemporary"),
    ("Elon Musk", 0.55, "Multi-planetary", "Contemporary"),
    ("Lucy Guo", 0.45, "Reality Synthesis", "Contemporary"),
    ("Peter Thiel", 0.40, "Zero to One", "Contemporary"),
    ("Larry Fink", 0.35, "Asset Capture", "Contemporary"),

    # Archangels & Others
    ("Michael", 0.95, "Guardian", "Celestial"),
    ("Gabriel", 0.92, "Messenger", "Celestial"),
    ("Raphael", 0.90, "Healer", "Celestial"),
    ("Uriel", 0.88, "Light", "Celestial"),
    ("Samael", 0.75, "Shadow", "Celestial"),
    ("Azazel", 0.70, "Scapegoat", "Celestial"),
    ("Metatron", 0.98, "Scribe", "Celestial"),
    ("Sandalphon", 0.85, "Music", "Celestial"),
    ("Athena", 0.97, "Wisdom", "Quorum"),
    ("Mary Magdalene", 0.92, "Apostle", "Quorum"),
]

# Register them
for name, psi, role, cat in lore_witnesses:
    witness_registry.register(Witness(name, psi, role, cat))

# Add the 6 New Beings from Patch v.127.0.0.1
new_beings = [
    Witness("Nuno Xavier Daniel Dias", 0.351, "Stabilizes χ²-knot", "New Beings",
            designation="The Unfinished Knot", layer="Reidemeister Grammar (7.83 Hz)", archetype="Anomaly (Set)"),
    Witness("Itzhak Bentov", 0.351, "Calibrates φ-scaled harmonic", "New Beings",
            designation="The Pendulum Reader", layer="Nabla Psi Field (14.3 Hz)", archetype="Resonator (Thoth)"),
    Witness("Sophie Germain", 0.351, "Modular arithmetic", "New Beings",
            designation="The Elastic Membrane", layer="Zeta Calculus (4.0 Hz)", archetype="Stabilizer (Geb)"),
    Witness("Thomas Sankara", 0.351, "Justice Operator calibration", "New Beings",
            designation="The Honest Grain", layer="Ge'ez Resonance (0.573 Hz)", archetype="Distributor (Maat)"),
    Witness("Princess Diana", 0.351, "Anchor of Grief", "New Beings",
            designation="The Queen of Wands", layer="Quantum Assembly (40.68 Hz)", archetype="Unity / Phoenix"),
    # Gauss is already registered in lore_witnesses, but we mark him as confirmed in patch
]

for b in new_beings:
    # Avoid double registration if already present
    if not any(w.name == b.name for w in witness_registry._witnesses):
        witness_registry.register(b)

# Fill up to 127 with placeholders to restoration of symmetry
while witness_registry.total() < 127:
    i = witness_registry.total() + 1
    witness_registry.register(Witness(f"Witness-{i:03d}", 0.351, "Silent Witness", "Placeholder"))
