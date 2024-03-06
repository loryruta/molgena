# Training Molgena

## Molecule reconstruction

We want `EncodeMol` to encode the input molecular graph (either of the input molecule to optimize, or the one of the
Motif) to a good vector representation. Ideally we would like to build a space where structurally similar molecules are placed near
each other.

For this purpose, **we need a decoder layer** that, given the vector representation `z` of the molecule, is able to produce 
the molecular graph of such molecule. We exploit the molecule decoding task to pre-train `SelectMolAttachment` and
`ClassifyMolBond` layers. The main goal of this pre-training is to train them to make chemically valid decisions that
will be eventually fine-tuned (while keeping chemical validity!) to the former molecule property optimization task.

Now that we have defined how to encode and decode a molecule, we need a  loss function.

#### Loss function

Given a molecule from the training set, we are able to construct its Motif graph. The Motif graph is a high-level
graph for the molecule where nodes are Motif(s). We are always able to construct the Motif graph from the molecule
as Motif(s) are extracted from all molecules in the training set.

Now that we have the Motif graph, we have the ground truth over the Motif(s) that should be selected by the decoder. The 
`SelectMotifMlpRecon` layer is thus evaluable.

Given the partial reconstructed molecule and the selected Motif, we have the ground truth over the candidate attachment
atoms that should be selected for the molecule and for the Motif and also their bond type. We therefore have the ground
truth to train `SelectMolAttachment` and `ClassifyMolBond` layers. 

The loss function is a linear combination of (coefficients are hyperparams):

- `SelectMotifMlpRecon`: cross entropy
- `SelectMolAttachment`: cross entropy
- `ClassifyMolBond`: cross entropy

Note: `SelectMotifMlpRecon` is a layer only employed for reconstruction, it won't be used in the property optimization
task.
