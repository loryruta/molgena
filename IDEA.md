Divide the molecule into Motifs: use the same technique described in HierVAE paper. It's also used in Microsoft's MoLeR
model.

## Ideas

### Reinforcement Learning

Have an agent which is trained to build a molecule from scratch or from an initial molecule in order to optimize a
property.

- State: the input molecule
- Action: extend the input molecule with a Motif or a single Atom (similar to MoLeR)
    - `PutAtomOrMotif`: connect an atom or a motif to the input molecule (where?)
    - `RemoveBond`: remove a bond such that the final molecule remains chemically valid
- Reward: molecule score. Should also consider that an action that currently isn't meaningful, could be meaningful for
  future actions.

`a = policy(m)`

The molecule is a graph, thus it needs to be encoded in order to be fed to the policy function.

How can we encode it?

GNN

Use Self-Attention and Positional-Encoding in order to have a memory over sequential Motif additions.

### Latent space

Build a latent space where similar molecules are placed near each other.

We assign a score `y(z)` to a few sample molecules, where `z` is the associated latent vector, in order to build the
search space.
Then we search for a latent vector `z*` that maximizes `y(z)`, optionally "near enough" to an input molecule (e.g. for
molecular optimization).

This approach is explored by:

- JT-VAE: uses Bayesian Optimization (and Gaussian Processes) to approximate `y(m)` and search
- MoLeR: uses Molecular Swarm Optimization (MSO)

## The training procedure

The training procedure is divided into 3 steps:





## References

- Gaussian Processes:
    - https://www.youtube.com/watch?v=UBDgSHPxVME
    - https://krasserm.github.io/2018/03/19/gaussian-processes/
- Bayesian Optimization: https://www.youtube.com/watch?v=M-NTkxfd7-8
- Molecular Swarm Optimization (MSO): https://chemrxiv.org/engage/chemrxiv/article-details/60c741269abda23bcaf8be35

