from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx

from core import IsotopomerEnumerator, IsotopomerResult


class IsotopomerVisualizer:

    def __init__(self, enumerator: IsotopomerEnumerator):
        self.enumerator = enumerator
        self.mol = enumerator.mol
        sns.set_style("whitegrid")
        self.mpl_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                           '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    def plot_cycle_index(self):
        cycle_index = self.enumerator.get_cycle_index()

        cycle_type_counts = defaultdict(lambda: defaultdict(int))

        for cycle_struct, count in cycle_index.items():
            for cycle_length in cycle_struct.cycle_lengths:
                cycle_type_counts[cycle_length]['count'] += count

        cycle_lengths = sorted(cycle_type_counts.keys())
        counts = [cycle_type_counts[cl]['count'] for cl in cycle_lengths]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(
            cycle_lengths, counts, color=self.mpl_colors[:len(cycle_lengths)],
            edgecolor='black', alpha=0.7)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Cycle Length', fontsize=12, fontweight='bold')
        ax.set_ylabel(
            'Frequency in Automorphism Group', fontsize=12, fontweight='bold')
        ax.set_title('Cycle Index of Automorphism Group\n' +
                     f'Group Order: {len(self.enumerator.automorphisms)} | ' +
                     f'Formula: {self.enumerator.get_formula()}',
                     fontsize=14, fontweight='bold')

        ax.set_xticks(cycle_lengths)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_degeneracy_histogram(self, results: List[IsotopomerResult]):
        degeneracies = [r.degeneracy for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(degeneracies, bins=20, color='steelblue',
                edgecolor='black', alpha=0.7)

        reduction_factor = sum(degeneracies) / len(results)

        ax.set_xlabel(
            'Degeneracy (Orbit Size)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Isotopomers', fontsize=12, fontweight='bold')
        ax.set_title('Degeneracy Distribution\n' +
                     f'Reduction Factor: {reduction_factor:.1f}',
                     fontsize=14, fontweight='bold')

        textstr = f'Total isotopomers: {len(results)}\n' + \
                  f'Symmetry group: |G| = {len(self.enumerator.automorphisms)}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        return fig

    def visualize_isotopomers(
            self,
            results: List[IsotopomerResult],
            max_show: int = 16,
            mols_per_row: int = 4
            ):

        mols_to_draw = []
        legends = []

        for i, result in enumerate(results[:max_show]):
            mol_copy = Chem.RWMol(self.mol)

            for atom_idx in result.atom_indices:
                atom = mol_copy.GetAtomWithIdx(atom_idx)
                atom.SetIsotope(result.isotope)

            mol_copy = mol_copy.GetMol()

            mols_to_draw.append(mol_copy)
            legends.append(f"#{i+1} | deg={result.degeneracy}")

        img = Draw.MolsToGridImage(
            mols_to_draw,
            molsPerRow=mols_per_row,
            subImgSize=(250, 250),
            legends=legends,
            highlightAtomLists=[
                list(r.atom_indices) for r in results[:max_show]],
            highlightBondLists=[[] for _ in results[:max_show]],
            returnPNG=False
        )

        return img

    def plot_symmetry_reduction_scaling(
            self,
            element: str,
            max_labels: int = None
            ):

        num_atoms = self.enumerator.get_num_atoms(element)
        if max_labels is None:
            max_labels = min(num_atoms, 8)

        label_counts = []
        unique_counts = []
        total_counts = []
        reduction_factors = []

        for n in range(1, max_labels + 1):
            results = self.enumerator.enumerate_isotopomers({element: n})
            unique = len(results)
            total = sum(r.degeneracy for r in results)

            label_counts.append(n)
            unique_counts.append(unique)
            total_counts.append(total)
            reduction_factors.append(total / unique if unique > 0 else 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(label_counts, total_counts, 'o-', label='Total labelings',
                 linewidth=2, markersize=8, color='steelblue')
        ax1.plot(label_counts, unique_counts, 's-', label='Unique isotopomers',
                 linewidth=2, markersize=8, color='orangered')
        ax1.set_xlabel(
            f'Number of {element} Labels', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'Isotopomer Scaling: {self.enumerator.get_formula()}',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        ax2.plot(label_counts, reduction_factors, 'D-', linewidth=2,
                 markersize=8, color='darkgreen')
        ax2.axhline(y=len(self.enumerator.automorphisms), color='red',
                    linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Group order |G| '
                    f'= {len(self.enumerator.automorphisms)}')
        ax2.set_xlabel(
            f'Number of {element} Labels', fontsize=12, fontweight='bold')
        ax2.set_ylabel(
            'Reduction Factor (Total/Unique)', fontsize=12, fontweight='bold')
        ax2.set_title(
            'Symmetry Compression Efficiency', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cycle_graph(
            self, layout='kamada_kawai',
            use_conjugacy_colors=True
            ):
        """
        Draw Cayley graph of automorphism group

        layout options:
        - 'kamada_kawai': Force-directed, minimizes edge length
            variance (deterministic)
        - 'spectral': Based on graph Laplacian eigenvectors (deterministic)
        - 'shell': Concentric shells by distance from identity
        - 'circular': Nodes on circle (works well for cyclic/dihedral groups)
        """
        automorphisms = self.enumerator.automorphisms
        n = len(automorphisms)

        # Generators: first few non-identity automorphisms
        generators = [a for a in automorphisms[1:min(4, n)]]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Build Cayley graph edges
        for i in range(n):
            for g in generators:
                composed = {
                    k: g[automorphisms[i][k]] for k in automorphisms[i]
                    }
                for j, aut in enumerate(automorphisms):
                    if aut == composed:
                        G.add_edge(i, j)
                        break

        # Layout selection
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G, scale=2)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G, scale=2)
        elif layout == 'shell':
            # Group by distance from identity (node 0)
            shells = []
            distances = nx.single_source_shortest_path_length(G, 0)
            max_dist = max(distances.values())
            for d in range(max_dist + 1):
                shell = [node for node, dist in distances.items() if dist == d]
                if shell:
                    shells.append(shell)
            pos = nx.shell_layout(G, nlist=shells, scale=2)
        elif layout == 'circular':
            pos = nx.circular_layout(G, scale=2)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Color by conjugacy class
        if use_conjugacy_colors:
            cycle_types = {}
            for i, aut in enumerate(automorphisms):
                cycle_struct = self.enumerator._permutation_to_cycles(aut)
                cycle_types[i] = cycle_struct

            unique_types = sorted(set(cycle_types.values()),
                                  key=lambda c: (
                                      len(c.cycle_lengths), c.cycle_lengths))
            color_map = {ct: self.mpl_colors[i % len(self.mpl_colors)]
                         for i, ct in enumerate(unique_types)}
            node_colors = [color_map[cycle_types[i]] for i in range(n)]
        else:
            node_colors = 'lightblue'

        fig, ax = plt.subplots(figsize=(14, 12))

        # Draw with padding to prevent node overlap
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=600, ax=ax, alpha=0.85,
                               edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.4, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

        # Legend
        if use_conjugacy_colors:
            cycle_index = self.enumerator.get_cycle_index()
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map[ct], markersize=12,
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=f'{ct.cycle_lengths} (n={cycle_index[ct]})')
                for ct in unique_types
            ]
            ax.legend(handles=legend_elements, loc='upper left',
                      fontsize=10, framealpha=0.9)

        ax.set_title(f'Cycle Graph of Automorphism Group ({layout} layout)\n'
                     f'Order: {n} | Formula: {self.enumerator.get_formula()}',
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.margins(0.15)  # Add margin to prevent clipping
        plt.tight_layout()
        return fig
