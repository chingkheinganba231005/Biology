# Cells to Code

**Biology from zero to computational biology**: molecular biology, bioinformatics, omics, machine learning, simulation, drug discovery and cancer research, with interactive lessons and end-to-end projects on real data.

**Start here:** open [`index.html`](index.html) in a browser, or the GitHub Pages site of this repository.

## Part 0 · Biology from zero

No biology assumed. From atoms and cells to DNA, proteins, genetics, the immune system, lab techniques, cancer and how drugs work: everything the computational fields build on, explained simply, with interactive models and a little Python.

| # | Chapter | What you learn |
|---|---|---|
| P1 | [Life, scale and the cell](prereq/01-life-and-cells/) | What living things are made of, how big everything is, and a tour of the cell as a factory. |
| P2 | [The chemistry of life](prereq/02-chemistry/) | Atoms, bonds, water, pH, energy, and the four families of molecules every cell is built from. |
| P3 | [DNA and genomes](prereq/03-dna/) | The double helix, base pairing, chromosomes, how DNA is copied, and what a genome really contains. |
| P4 | [From gene to protein](prereq/04-central-dogma/) | Transcription, splicing, the genetic code and translation: how a DNA sequence becomes a working protein. |
| P5 | [Proteins: shape is function](prereq/05-proteins/) | Amino acids, folding, the four levels of structure, enzymes and how fast they work. |
| P6 | [Gene regulation and epigenetics](prereq/06-regulation/) | How a cell decides which genes to switch on: promoters, transcription factors, chromatin and cell identity. |
| P7 | [The cell cycle, division and death](prereq/07-cell-cycle/) | Mitosis, meiosis, checkpoints, p53 and apoptosis: the machinery cancer breaks. |
| P8 | [Genetics, mutation and evolution](prereq/08-genetics-evolution/) | Inheritance, alleles, mutations, populations, natural selection and the tree of life. |
| P9 | Signalling and metabolism *(coming)* | How cells sense and respond (receptors, kinase cascades) and how they make and use energy. |
| P10 | The immune system *(coming)* | Innate and adaptive immunity, antibodies, T cells and the basis of vaccines and immunotherapy. |
| P11 | The lab toolkit *(coming)* | PCR, gels, cloning, sequencing, CRISPR, microscopy and assays: where biological data comes from. |
| P12 | Cancer biology *(coming)* | Oncogenes, tumour suppressors, the hallmarks of cancer, tumour evolution and how cancer is treated. |
| P13 | How drugs work *(coming)* | Targets, binding, dose–response, potency and selectivity, ADME, and the drug-discovery pipeline. |

## Part 1 · Bioinformatics

Biological sequences as data: reading genomes, aligning sequences, building evolutionary trees, assembling genomes and finding variants, all from scratch on real data.

| # | Chapter | What you learn |
|---|---|---|
| 1 | Sequences as data *(coming)* | FASTA, GC content, k-mers, reading frames and gene finding in a real virus genome. |
| 2 | Sequence alignment from scratch *(coming)* | Dynamic programming, Needleman–Wunsch and Smith–Waterman, substitution matrices, and how BLAST searches billions of letters. |
| 3 | Phylogenetics: trees of life *(coming)* | Evolutionary distances, UPGMA and neighbour joining, and a tree of real viral genomes. |
| 4 | Genome assembly and read mapping *(coming)* | Sequencing reads, quality scores, de Bruijn graph assembly and fast read mapping with k-mer indexes. |
| 5 | Variants and population genomics *(coming)* | Calling variants, allele frequencies, Hardy–Weinberg, linkage and a genome-wide association study. |

## Part 2 · Omics and machine learning

Measuring thousands of genes at once and learning from it: RNA-seq, single-cell data, cancer classification and deep learning on DNA and proteins.

| # | Chapter | What you learn |
|---|---|---|
| 6 | Transcriptomics and differential expression *(coming)* | RNA-seq counts, normalisation, statistical testing with false-discovery control, on real tumour data. |
| 7 | Single-cell RNA-seq *(coming)* | Quality control, PCA, neighbourhood graphs, clustering and cell-type annotation of 2,700 real blood cells. |
| 8 | Machine learning for cancer genomics *(coming)* | Classifying tumours from gene expression honestly: high dimensions, feature selection, leakage and interpretation. |
| 9 | Deep learning on DNA *(coming)* | One-hot sequences, convolutional networks that learn binding motifs, and how to read what they learned. |
| 10 | Protein machine learning and AlphaFold *(coming)* | Sequence embeddings, coevolution, contact prediction and how AlphaFold and protein language models work. |

## Part 3 · Structure, chemistry and drug discovery

Molecules in 3D, physics-based simulation and the computational side of finding new medicines.

| # | Chapter | What you learn |
|---|---|---|
| 11 | Protein structure *(coming)* | PDB files, coordinates, contacts, RMSD and superposition, binding pockets and AlphaFold confidence. |
| 12 | Molecular dynamics from scratch *(coming)* | Force fields, Verlet integration, temperature, and a toy protein that folds on a lattice. |
| 13 | Cheminformatics *(coming)* | Molecules as graphs: SMILES, descriptors, fingerprints, drug-likeness and similarity search with RDKit. |
| 14 | QSAR and virtual screening *(coming)* | Predicting activity against a cancer target from real ChEMBL data, with honest scaffold splits. |
| 15 | Docking and binding *(coming)* | Scoring functions, a docking search from scratch, and what binding free energy means. |
| 16 | Pharmacokinetics and dosing *(coming)* | Compartment models, half-life, dosing regimens and dose–response, as differential equations. |

## Part 4 · Systems biology and simulation

Cells and tissues as dynamical systems: gene circuits, noise, and the evolution of tumours, simulated.

| # | Chapter | What you learn |
|---|---|---|
| 17 | Gene circuits as dynamical systems *(coming)* | Differential equations for gene expression: switches, oscillators and feedback. |
| 18 | Noise and stochastic simulation *(coming)* | Why identical cells differ: the Gillespie algorithm and bursting gene expression. |
| 19 | Tumour growth and cancer evolution *(coming)* | Growth laws, agent-based tumours, clonal evolution and how drug resistance emerges. |

## Part 5 · Cancer research and capstones

Real cancer genomics and clinical data, then end-to-end projects that use the whole course.

| # | Chapter | What you learn |
|---|---|---|
| 20 | Cancer genomics and mutational signatures *(coming)* | Driver genes, mutation spectra and signature extraction with non-negative matrix factorisation. |
| 21 | Survival analysis *(coming)* | Kaplan–Meier curves, log-rank tests and Cox models on real patient cohorts. |
| 22 | Capstone: a drug-discovery campaign *(coming)* | From a cancer target to ranked candidate molecules: data, models, screening, filtering and a report. |
| 23 | Capstone: a precision-oncology analysis *(coming)* | From a tumour cohort's mutations and expression to subtypes, biomarkers and outcomes. |

## Running the notebooks

```bash
python3 -m pip install -U jupyter numpy pandas matplotlib scipy scikit-learn biopython rdkit requests
jupyter notebook
```

Each chapter folder contains `index.html` (the guide), a Jupyter notebook that reproduces every number and figure, `images/`, `datasets/` and `animations/` (interactive playgrounds). Guides and notebooks are generated from a single literate source, `chapter.md`, with `python3 tools/build.py <chapter-folder>`; this page and the home page with `python3 tools/hub.py`.

*Educational material. Nothing here is medical advice.*
