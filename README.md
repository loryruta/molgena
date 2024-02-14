# MOLGENA (Molecular genarator)

Project developed for the exam of the course AI for Bioinformatics.

### Chemical properties to optimize:

- logp (Octanol-water Partition Coefficient): measures the solubility and synthetic accessibility of a compound
- QED (Quantitative Estimate of Drug-likeness)
- SA (Synthetic Accessibility): how hard/easy is to synthetize the molecule
- MW (Molecular Weight)

### Metrics

- Frechet ChemNet Distance (FCD, used in MoLeR): measure how much realistic are generated molecules
- Tanimoto similarity coefficient ([Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)): used to
  measure [similarity](https://en.wikipedia.org/wiki/Chemical_similarity) between two molecules

# Concepts

- Molecular fingerprint: encode properties of small molecules and assess their similarities computationally through bit
  string comparisons
- Tanimoto similarity: the ratio of common structures divided by the combined structures
    - https://www.youtube.com/watch?v=3qzZbaUzo9M

- Variational Inference & ELBO: https://www.youtube.com/watch?v=HxQ94L8n0vU

### Molecular fingerprint

The molecular fingerprint is a bit-vector that encodes the structural features of a molecule and it's used, for example,
to compare molecular similarity.
There are many fingerprint encoding algorithms available, the "best one" strongly depends on the dataset and on the
task.

Most encoding algorithms extract features from the molecule, hash them, and use the hash to compute the bit-vector
position to set.

###### References

- https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf

### Tanimoto similarity

Tanimoto similarity, also known as Jaccard index, computes the similarity between two molecules.

Given the Morgan fingerprint of the two molecules, it's evaluated as:

```
sim(fp1, fp2) = intersect(fp1, fp2) / union(fp1, fp2)
```

###### References

- Jaccard index: https://en.wikipedia.org/wiki/Jaccard_index

### Kekule structure

Same as Lewis structure for representing molecule geometry but without lone pairs and formal charges (electrons weren't
discovered yet!).

###### References

- https://www.chem.ucla.edu/~harding/IGOC/K/kekule_structure.html

# Task we're interested in:

- **Constrained molecule optimization**: very useful in drug discovery, the generation of new drugs usually starts with
  known molecules (such as existing drugs). The objective of this task is to generate a novel molecule, starting from an
  initial molecule, that improves its chemical properties.

# Benchmarks

### Guacamol (https://github.com/BenevolentAI/guacamol)

Guacamol is an evaluation framework based on a suite of standardised benchmarks for de-novo molecular design.
It's thought to assess both classical methods and neural model -based methods.

Two benchmarks:

- `assess_distribution_learning`: ability to generate molecules similar to those in a training set

```py
@abstractmethod
def generate(self, number_samples: int) -> List[str]
    """
    Samples SMILES strings from a molecule generator.

    Args:
        number_samples: number of molecules to generate

    Returns:
        A list of SMILES strings.
    """
    pass
```

- `goal_directed_generator`: ability to generate molecules that achieve a high score for a given scoring function

```py
@abstractmethod
def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                 starting_population: Optional[List[str]] = None) -> List[str]:
    """
    Given an objective function, generate molecules that score as high as possible.

    Args:
        scoring_function: scoring function
        number_molecules: number of molecules to generate
        starting_population: molecules to start the optimization from (optional)

    Returns:
        A list of SMILES strings for the generated molecules.
    """
    pass
```

Implementation examples can be found at https://github.com/BenevolentAI/guacamol_baselines.

###### References:

- https://github.com/BenevolentAI/guacamol?tab=readme-ov-file#benchmarking-models

### MOSES (https://github.com/molecularsets/moses)

A dataset obtained from filtering ZINC, used for benchmarking. Can be used to assess the overall quality of generated
molecules.

Measured metrics:

- Uniqueness (↑)
- Validity (↑)
- Fragment similarity (Frag) (↑): consine distance over vector of fragment frequencies between generated and test set
- Scaffold similarity (Scaff) (↑): cosine distance over vector of scaffold frequencies between generated and test set
- Nearest neighbor similarity (SNN) (↑), : average similarity of generated molecule with the nearest molecule from the
  test set
- Internal diversity (IntDiv) (↑): pairwise similarity of generated molecules
- Fréchet ChemNet Distance (FCD) (↓): difference in distributions of last layer activations of ChemNet
- Novelty (↑): fraction of unique valid generated molecules not present in the training set

_TODO (UNDERSTAND): To compare molecular properties: Wasserstein-1 distance between distributions of molecules in the
generated and test set_

###### References

- https://github.com/molecularsets/moses?tab=readme-ov-file#metrics
