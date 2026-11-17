# NucleasePrism Architecture & Mathematical Invariants

## Design Philosophy

NucleasePrism separates biophysical modeling into decoupled, strictly typed layers so each subsystem can be verified independently against conservation laws, brute-force sequence oracles, and parametric calibration targets without external C/CUDA dependencies.

## Data Flow & Subsystem Boundaries

1. **Nuclease Geometry (`nucleases.py`)**: Defines PAM IUPAC patterns, non-canonical PAM licensing weights, 5'- vs 3'-PAM strand orientations, and directional R-loop zipping coordinate maps.
2. **Sequence Mechanics (`sequence.py`)**: Extracts forward/reverse protospacer candidates, computes Zuker/Nussinov spacer-scaffold RNA hairpin MFE, detects Pol-III `TTTT` terminators, and scans flanking microhomology direct repeats.
3. **Thermodynamics (`thermodynamics.py`)**: Evaluates Sugimoto RNA:DNA and SantaLucia DNA:DNA nearest-neighbor free energies, mismatch wobble vs clash penalties, and solves the tridiagonal continuous-time Markov R-loop master equation $\mathbf{Q}^\top \boldsymbol{\tau} = -\mathbf{e}_0$.
4. **Genome Indexing (`indexing.py`)**: Builds a multi-contig suffix-array BWT/FM-index with sampled `Occ` checkpoints, $k$-mismatch branch-and-bound search, 2-bit Myers XOR Hamming comparison, and seed-anchored 1-nt DNA/RNA bulge alignment.
5. **Scoring & Repair (`scoring.py`, `repair.py`)**: Integrates R-loop cleavage probability with positional sequence features, CFD/MIT off-target aggregation, base/prime editing windows, and RK45 DSB repair pathway ODEs.
6. **Haplotype & Screens (`haplotype.py`, `screens.py`)**: Quantifies REF vs ALT allele-specific cleavage discrimination, multiplex greedy set cover, and MAGeCK-style Negative Binomial RRA pooled screen ranking.
7. **Calibration & Telemetry (`calibration.py`, `pipeline.py`, `visualizer.py`, `cli.py`)**: Provides bounded Nelder-Mead parameter estimation, Saltelli Sobol sensitivity indices, Pareto frontier ranking, JSON/CSV serialization, and strict 7-bit ASCII terminal plots.

## Verified Mathematical Invariants

- **Markov Residence Conservation**: Stationary quasi-steady occupancy satisfies $\sum_{s=0}^{L} p_s = 1.0$.
- **ODE Mass Conservation**: DSB repair fractions satisfy $U(t) + N(t) + M(t) + H(t) = 1.0$ across all $t \in [0, 24]\text{ h}$.
- **Exact FM-Index Equivalence**: $k$-mismatch search returns the exact same coordinate set as a 2-bit bit-parallel exhaustive scan.
- **Strict 7-Bit ASCII Output**: All CLI and visualizer strings satisfy `ord(c) < 128` for Windows `cp1252` compatibility.
