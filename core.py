from typing import Dict, Tuple, List, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools
from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
from networkx.algorithms import isomorphism


@dataclass
class IsotopomerResult:
    isotopomer_id: int
    atom_indices: Tuple[int, ...]
    element: str
    isotope: int
    degeneracy: int
    inchi: str
    smiles: str


@dataclass
class CycleStructure:
    cycle_lengths: Tuple[int, ...]
    num_fixed_points: int

    def __hash__(self):
        return hash(self.cycle_lengths)


class IsotopomerEnumerator:

    def __init__(self, molecule_input: str, input_type: str = "smiles"):
        self.input_type = input_type
        self.mol = self._parse_molecule(molecule_input, input_type)
        self.mol = Chem.AddHs(self.mol)
        self.graph = self._build_graph()
        self.automorphisms = self._compute_automorphisms()
        self._cycle_index_cache = {}

    def _parse_molecule(self, molecule_input: str, input_type: str):
        if input_type == "smiles":
            mol = Chem.MolFromSmiles(molecule_input)
        elif input_type == "inchi":
            mol = Chem.MolFromInchi(molecule_input)
        else:
            mol = Chem.MolFromSmiles(molecule_input)
        return mol

    def _build_graph(self):
        G = nx.Graph()

        for atom in self.mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization())
            )

        for bond in self.mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=str(bond.GetBondType())
            )

        return G

    def _node_match(self, n1: dict, n2: dict):
        return (n1['element'] == n2['element'] and
                n1['aromatic'] == n2['aromatic'] and
                n1['charge'] == n2['charge'] and
                n1['hybridization'] == n2['hybridization'])

    def _edge_match(self, e1: dict, e2: dict):
        return e1['bond_type'] == e2['bond_type']

    def _compute_automorphisms(self):
        GM = isomorphism.GraphMatcher(
            self.graph,
            self.graph,
            node_match=self._node_match,
            edge_match=self._edge_match
        )

        automorphisms = list(GM.isomorphisms_iter())
        n_atoms = self.mol.GetNumAtoms()
        identity = {i: i for i in range(n_atoms)}

        if identity not in automorphisms:
            automorphisms.insert(0, identity)

        return automorphisms

    def _permutation_to_cycles(self, perm: Dict[int, int]):
        visited = set()
        cycles = []

        for start in perm.keys():
            if start in visited:
                continue

            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = perm[current]

            if len(cycle) > 0:
                cycles.append(len(cycle))

        cycles = tuple(sorted(cycles, reverse=True))
        num_fixed = sum(1 for c in cycles if c == 1)

        return CycleStructure(cycles, num_fixed)

    def get_cycle_index(self):
        if self._cycle_index_cache:
            return self._cycle_index_cache

        cycle_counts = defaultdict(int)

        for automorphism in self.automorphisms:
            cycle_struct = self._permutation_to_cycles(automorphism)
            cycle_counts[cycle_struct] += 1

        self._cycle_index_cache = dict(cycle_counts)
        return self._cycle_index_cache

    def _compute_orbit_generic(self, labeling):
        orbit = set()

        if isinstance(labeling, frozenset):
            for automorphism in self.automorphisms:
                transformed = frozenset(
                    automorphism[atom] for atom in labeling)
                orbit.add(transformed)
        else:
            for automorphism in self.automorphisms:
                transformed = {}
                for element, atom_set in labeling.items():
                    transformed[element] = frozenset(
                        automorphism[atom] for atom in atom_set
                    )

                transformed_key = frozenset(
                    (elem, frozenset(atoms))
                    for elem, atoms in transformed.items()
                )
                orbit.add(transformed_key)

        return orbit

    def enumerate_isotopomers(
        self,
        label_spec: Dict[str, int],
        isotope_map: Dict[str, int] = None
    ):
        if isotope_map is None:
            isotope_map = {'C': 13, 'H': 2, 'N': 15, 'O': 18, 'S': 34}

        if len(label_spec) > 1:
            return self._enumerate_multi_element(label_spec, isotope_map)

        element = list(label_spec.keys())[0]
        num_labels = label_spec[element]
        isotope = isotope_map.get(element, 13)

        element_atoms = [atom.GetIdx() for atom in self.mol.GetAtoms()
                         if atom.GetSymbol() == element]

        all_labelings = list(itertools.combinations(element_atoms, num_labels))

        results = []
        orbits_found = set()
        isotopomer_id = 0

        for labeling in all_labelings:
            labeling_set = frozenset(labeling)

            if labeling_set in orbits_found:
                continue

            orbit = self._compute_orbit_generic(labeling_set)
            orbits_found.update(orbit)
            degeneracy = len(orbit)

            labeled_mol = Chem.RWMol(self.mol)
            for atom_idx in labeling:
                atom = labeled_mol.GetAtomWithIdx(atom_idx)
                atom.SetIsotope(isotope)
            labeled_mol = labeled_mol.GetMol()

            inchi = Chem.MolToInchi(labeled_mol)
            smiles = Chem.MolToSmiles(labeled_mol)

            result = IsotopomerResult(
                isotopomer_id=isotopomer_id,
                atom_indices=tuple(sorted(labeling)),
                element=element,
                isotope=isotope,
                degeneracy=degeneracy,
                inchi=inchi,
                smiles=smiles
            )

            results.append(result)
            isotopomer_id += 1

        return results

    def _enumerate_multi_element(
            self,
            label_spec: Dict[str, int],
            isotope_map: Dict[str, int]
            ):
        element_atoms = {}
        for element in label_spec.keys():
            element_atoms[element] = [
                atom.GetIdx() for atom in self.mol.GetAtoms()
                if atom.GetSymbol() == element
            ]

        element_combinations = {}
        for element, num_labels in label_spec.items():
            element_combinations[element] = list(
                itertools.combinations(element_atoms[element], num_labels)
            )

        elements = list(label_spec.keys())
        all_combo_lists = [element_combinations[e] for e in elements]
        all_multi_labelings = list(itertools.product(*all_combo_lists))

        multi_labelings = []
        for combo_tuple in all_multi_labelings:
            labeling = {}
            for i, element in enumerate(elements):
                labeling[element] = frozenset(combo_tuple[i])
            multi_labelings.append(labeling)

        results = []
        orbits_found = set()
        isotopomer_id = 0

        for labeling in multi_labelings:
            labeling_key = frozenset(
                (elem, frozenset(atoms))
                for elem, atoms in labeling.items()
            )

            if labeling_key in orbits_found:
                continue

            orbit = self._compute_orbit_generic(labeling)
            orbits_found.update(orbit)
            degeneracy = len(orbit)

            labeled_mol = Chem.RWMol(self.mol)
            for element, atom_indices in labeling.items():
                isotope = isotope_map.get(element, 13)
                for atom_idx in atom_indices:
                    atom = labeled_mol.GetAtomWithIdx(atom_idx)
                    atom.SetIsotope(isotope)
            labeled_mol = labeled_mol.GetMol()

            inchi = Chem.MolToInchi(labeled_mol)
            smiles = Chem.MolToSmiles(labeled_mol)

            all_atom_indices = []
            for element in sorted(labeling.keys()):
                all_atom_indices.extend(sorted(labeling[element]))
            all_atom_indices = tuple(sorted(all_atom_indices))

            result = IsotopomerResult(
                isotopomer_id=isotopomer_id,
                atom_indices=all_atom_indices,
                element="+".join(sorted(elements)),
                isotope=isotope_map.get(elements[0], 13),
                degeneracy=degeneracy,
                inchi=inchi,
                smiles=smiles
            )

            results.append(result)
            isotopomer_id += 1

        return results

    def get_formula(self):
        return Descriptors.rdMolDescriptors.CalcMolFormula(self.mol)

    def get_num_atoms(self, element: str = None):
        if element is None:
            return self.mol.GetNumAtoms()
        return sum(
            1 for atom in self.mol.GetAtoms() if atom.GetSymbol() == element)

    def get_symmetry_group_size(self):
        return len(self.automorphisms)


def quick_count(smiles: str, label_spec: Dict[str, int]) -> int:
    enumerator = IsotopomerEnumerator(smiles, "smiles")
    results = enumerator.enumerate_isotopomers(label_spec)
    return len(results)


@dataclass
class IsotopomerInChIData:
    """
    Data structure for InChI-based validation of isotopomer enumeration.

    Attributes:
        label_spec: Dictionary specifying number of labels per element
        total_labelings: Total combinatorial labelings C(n,k)
        unique_orbits: Number of unique symmetry orbits (from Burnside)
        unique_inchis: Number of unique InChI strings
        inchi_to_labelings: Map from InChI to list of atom index tuples
        orbit_to_inchi: Map from orbit representative to InChI
        validation_passed: Whether unique_orbits == unique_inchis
        degeneracy_sum: Sum of all orbit degeneracies
    """
    label_spec: Dict[str, int]
    total_labelings: int
    unique_orbits: int
    unique_inchis: int
    inchi_to_labelings: Dict[str, List[Tuple[int, ...]]]
    orbit_to_inchi: Dict[Tuple[int, ...], str]
    validation_passed: bool
    degeneracy_sum: int
    
    def get_summary(self) -> str:
        """Generate human-readable validation summary."""
        lines = [
            f"Label specification: {self.label_spec}",
            f"Total combinatorial labelings: {self.total_labelings}",
            f"Unique orbits (Burnside): {self.unique_orbits}",
            f"Unique InChI strings: {self.unique_inchis}",
            f"Degeneracy sum: {self.degeneracy_sum}",
            f"Validation: {'PASSED' if self.validation_passed else 'FAILED'}",
        ]
        
        if not self.validation_passed:
            lines.append(f"  ERROR: Orbit count ({self.unique_orbits}) != InChI count ({self.unique_inchis})")
        
        return "\n".join(lines)


def enumerate_all_isotopomer_inchis(
    enumerator,
    label_spec: Dict[str, int],
    isotope_map: Dict[str, int] = None
) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Generate InChI strings for ALL combinatorial labelings (not just unique orbits).
    
    This brute-force enumeration provides ground truth for validation:
    each symmetry orbit should produce exactly one unique InChI string.
    
    Args:
        enumerator: IsotopomerEnumerator instance
        label_spec: Dictionary mapping elements to number of labels
        isotope_map: Dictionary mapping elements to isotope mass numbers
        
    Returns:
        Dictionary mapping InChI strings to lists of atom index tuples that
        produce that InChI. For symmetric molecules, multiple labelings map
        to the same InChI (they are in the same orbit).
        
    Example:
        For benzene with 2 13C labels:
        - Total labelings: C(6,2) = 15
        - Unique InChIs: 3 (due to D6h symmetry)
        - Some InChIs will have 6 labelings each (high degeneracy orbits)
    """
    if isotope_map is None:
        isotope_map = {'C': 13, 'H': 2, 'N': 15, 'O': 18, 'S': 34}
    
    inchi_to_labelings = defaultdict(list)
    
    # Handle single-element case
    if len(label_spec) == 1:
        element = list(label_spec.keys())[0]
        num_labels = label_spec[element]
        isotope = isotope_map.get(element, 13)
        
        element_atoms = [
            atom.GetIdx() for atom in enumerator.mol.GetAtoms()
            if atom.GetSymbol() == element
        ]
        
        all_labelings = list(itertools.combinations(element_atoms, num_labels))
        
        for labeling in all_labelings:
            labeled_mol = Chem.RWMol(enumerator.mol)
            for atom_idx in labeling:
                atom = labeled_mol.GetAtomWithIdx(atom_idx)
                atom.SetIsotope(isotope)
            labeled_mol = labeled_mol.GetMol()
            
            inchi = Chem.MolToInchi(labeled_mol)
            inchi_to_labelings[inchi].append(tuple(sorted(labeling)))
    
    # Handle multi-element case
    else:
        element_atoms = {}
        for element in label_spec.keys():
            element_atoms[element] = [
                atom.GetIdx() for atom in enumerator.mol.GetAtoms()
                if atom.GetSymbol() == element
            ]
        
        element_combinations = {}
        for element, num_labels in label_spec.items():
            element_combinations[element] = list(
                itertools.combinations(element_atoms[element], num_labels)
            )
        
        elements = list(label_spec.keys())
        all_combo_lists = [element_combinations[e] for e in elements]
        all_multi_labelings = list(itertools.product(*all_combo_lists))
        
        for combo_tuple in all_multi_labelings:
            labeled_mol = Chem.RWMol(enumerator.mol)
            
            all_atom_indices = []
            for i, element in enumerate(elements):
                isotope = isotope_map.get(element, 13)
                for atom_idx in combo_tuple[i]:
                    atom = labeled_mol.GetAtomWithIdx(atom_idx)
                    atom.SetIsotope(isotope)
                    all_atom_indices.append(atom_idx)
            
            labeled_mol = labeled_mol.GetMol()
            inchi = Chem.MolToInchi(labeled_mol)
            inchi_to_labelings[inchi].append(tuple(sorted(all_atom_indices)))
    
    return dict(inchi_to_labelings)


def get_unique_inchis(
    enumerator,
    label_spec: Dict[str, int],
    isotope_map: Dict[str, int] = None
) -> Set[str]:
    """
    Get the set of unique InChI strings for a given labeling specification.
    
    This provides ground truth for validation: the number of unique InChI
    strings should exactly equal the number of symmetry orbits computed
    via Burnside's Lemma.
    
    Args:
        enumerator: IsotopomerEnumerator instance
        label_spec: Dictionary mapping elements to number of labels
        isotope_map: Dictionary mapping elements to isotope mass numbers
        
    Returns:
        Set of unique InChI strings
    """
    inchi_map = enumerate_all_isotopomer_inchis(enumerator, label_spec, isotope_map)
    return set(inchi_map.keys())


def verify_orbit_partition(
    enumerator,
    results: List,
    label_spec: Dict[str, int],
    isotope_map: Dict[str, int] = None
) -> Dict[str, any]:
    """
    Verify that Burnside enumeration correctly partitions labelings into orbits.
    
    This function performs comprehensive validation:
    1. Each orbit should produce exactly one unique InChI
    2. Sum of orbit degeneracies should equal total combinatorial count
    3. Number of orbits should equal number of unique InChIs
    
    Args:
        enumerator: IsotopomerEnumerator instance
        results: List of IsotopomerResult from enumerate_isotopomers
        label_spec: Dictionary mapping elements to number of labels
        isotope_map: Dictionary mapping elements to isotope mass numbers
        
    Returns:
        Dictionary containing validation results with keys:
            - 'valid': Boolean indicating if validation passed
            - 'num_orbits': Number of unique orbits from Burnside
            - 'num_unique_inchis': Number of unique InChI strings
            - 'total_labelings': Total combinatorial count
            - 'degeneracy_sum': Sum of all degeneracies
            - 'orbit_inchis': List of InChIs for each orbit
            - 'inchi_degeneracies': Dict mapping InChI to degeneracy from brute force
            - 'errors': List of error messages if validation failed
    """
    if isotope_map is None:
        isotope_map = {'C': 13, 'H': 2, 'N': 15, 'O': 18, 'S': 34}
    
    errors = []
    
    # Get InChI strings from Burnside enumeration results
    orbit_inchis = [r.inchi for r in results]
    num_orbits = len(results)
    
    # Get unique InChIs from brute-force enumeration
    inchi_map = enumerate_all_isotopomer_inchis(enumerator, label_spec, isotope_map)
    num_unique_inchis = len(inchi_map)
    
    # Calculate expected total from combinatorics
    if len(label_spec) == 1:
        element = list(label_spec.keys())[0]
        num_labels = label_spec[element]
        n_atoms = enumerator.get_num_atoms(element)
        
        from math import comb
        total_labelings = comb(n_atoms, num_labels)
    else:
        # Multi-element: product of individual combinations
        from math import comb
        total_labelings = 1
        for element, num_labels in label_spec.items():
            n_atoms = enumerator.get_num_atoms(element)
            total_labelings *= comb(n_atoms, num_labels)
    
    # Sum of degeneracies from Burnside results
    degeneracy_sum = sum(r.degeneracy for r in results)
    
    # Validation checks
    if num_orbits != num_unique_inchis:
        errors.append(
            f"Orbit count mismatch: Burnside gave {num_orbits} orbits, "
            f"but InChI canonicalization found {num_unique_inchis} unique structures"
        )
    
    if degeneracy_sum != total_labelings:
        errors.append(
            f"Degeneracy sum mismatch: Sum of degeneracies is {degeneracy_sum}, "
            f"but expected {total_labelings} from combinatorics"
        )
    
    # Check that each orbit InChI appears in brute-force enumeration
    orbit_inchi_set = set(orbit_inchis)
    brute_force_inchi_set = set(inchi_map.keys())
    
    if orbit_inchi_set != brute_force_inchi_set:
        missing_in_brute = orbit_inchi_set - brute_force_inchi_set
        missing_in_orbits = brute_force_inchi_set - orbit_inchi_set
        
        if missing_in_brute:
            errors.append(f"Orbits contain {len(missing_in_brute)} InChIs not found in brute-force enumeration")
        if missing_in_orbits:
            errors.append(f"Brute-force found {len(missing_in_orbits)} InChIs not in orbit enumeration")
    
    # Get degeneracies from brute-force enumeration
    inchi_degeneracies = {inchi: len(labelings) for inchi, labelings in inchi_map.items()}
    
    return {
        'valid': len(errors) == 0,
        'num_orbits': num_orbits,
        'num_unique_inchis': num_unique_inchis,
        'total_labelings': total_labelings,
        'degeneracy_sum': degeneracy_sum,
        'orbit_inchis': orbit_inchis,
        'inchi_degeneracies': inchi_degeneracies,
        'errors': errors
    }


def generate_inchi_data(
    enumerator,
    label_spec: Dict[str, int],
    isotope_map: Dict[str, int] = None
) -> IsotopomerInChIData:
    """
    Generate comprehensive InChI validation data for isotopomer enumeration.
    
    This is the high-level function for complete validation analysis,
    combining Burnside enumeration with InChI canonicalization verification.
    
    Args:
        enumerator: IsotopomerEnumerator instance
        label_spec: Dictionary mapping elements to number of labels
        isotope_map: Dictionary mapping elements to isotope mass numbers
        
    Returns:
        IsotopomerInChIData object containing comparison results
    """
    if isotope_map is None:
        isotope_map = {'C': 13, 'H': 2, 'N': 15, 'O': 18, 'S': 34}
    
    # Run Burnside enumeration
    results = enumerator.enumerate_isotopomers(label_spec, isotope_map)
    
    # Run brute-force InChI enumeration
    inchi_to_labelings = enumerate_all_isotopomer_inchis(enumerator, label_spec, isotope_map)
    
    # Build orbit to InChI mapping
    orbit_to_inchi = {r.atom_indices: r.inchi for r in results}
    
    # Calculate totals
    from math import comb
    if len(label_spec) == 1:
        element = list(label_spec.keys())[0]
        num_labels = label_spec[element]
        n_atoms = enumerator.get_num_atoms(element)
        total_labelings = comb(n_atoms, num_labels)
    else:
        total_labelings = 1
        for element, num_labels in label_spec.items():
            n_atoms = enumerator.get_num_atoms(element)
            total_labelings *= comb(n_atoms, num_labels)
    
    unique_orbits = len(results)
    unique_inchis = len(inchi_to_labelings)
    degeneracy_sum = sum(r.degeneracy for r in results)
    
    comparison_passed = (
        unique_orbits == unique_inchis and
        degeneracy_sum == total_labelings
    )
    
    return IsotopomerInChIData(
        label_spec=label_spec,
        total_labelings=total_labelings,
        unique_orbits=unique_orbits,
        unique_inchis=unique_inchis,
        inchi_to_labelings=inchi_to_labelings,
        orbit_to_inchi=orbit_to_inchi,
        comparison_passed=comparison_passed,
        degeneracy_sum=degeneracy_sum
    )