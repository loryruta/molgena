import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from rdkit import Chem
import rdkit.Chem.Draw
import numpy as np

from runtime_context import parse_runtime_context_from_cmdline, RuntimeContext
from mol_dataset import ZincDataset
from inference import ReconstructTask, ReconstructResult
from utils.chem_utils import smiles_to_image, calc_tanimoto_similarity


class ReconstructView:
    USE_TRAINING_SET = False
    NUM_MOLS = 4

    def __init__(self, context: RuntimeContext):
        if self.USE_TRAINING_SET:
            self._dataset = ZincDataset.training_set()
        else:
            self._dataset = ZincDataset.test_set()

        self._reconstruct_task = ReconstructTask.from_context(context)
        self._reconstruct_task.verbose = False

        fig, axs = plt.subplots(self.NUM_MOLS, 4)
        self._fig = fig
        self._axs = axs

        self._step = 0

    def _sample_tsmiles(self):
        self._tsmiles_list = self._dataset.df.sample(n=self.NUM_MOLS)['smiles'].tolist()

        # Reconstruct sampled tsmiles
        self._results: List[Tuple[List[str], ReconstructResult]] = []
        for tsmiles in self._tsmiles_list:
            self._results.append(self._reconstruct_task.run_reconstruct(tsmiles))

        self._max_step = max([len(psmiles_list) for psmiles_list, _ in self._results])

        # Calc Tanimoto similarities
        self._scores: List[List[int]] = []
        for tsmiles, (psmiles_list, _) in zip(self._tsmiles_list, self._results):
            scores = []
            for psmiles in psmiles_list:
                scores.append(calc_tanimoto_similarity(tsmiles, psmiles))
            self._scores.append(scores)

    def _redraw_canvas(self):
        def set_image_caption(ax, text: str, size: int = 8):
            ax.text(0.5, -0.1, text, size=size, ha='center', transform=ax.transAxes)

        self._fig.suptitle(f"Step: {self._step}")

        # Clear axes
        for ax in self._axs.flatten():
            ax.cla()

        for ax in self._axs[:, :3].flatten():
            ax.axis('off')

        self._axs[0, 0] \
            .text(0.5, 1.2, "Target molecule", transform=self._axs[0, 0].transAxes, ha='center', va='center')
        self._axs[0, 1] \
            .text(0.5, 1.2, "Partial molecule", transform=self._axs[0, 1].transAxes, ha='center', va='center')
        self._axs[0, 2] \
            .text(0.5, 1.2, "Result", transform=self._axs[0, 2].transAxes, ha='center', va='center')
        self._axs[0, 3] \
            .text(0.5, 1.2, "Tanimoto similarity", transform=self._axs[0, 3].transAxes, ha='center', va='center')

        for r in range(0, self.NUM_MOLS):
            tsmiles = self._tsmiles_list[r]
            self._axs[r, 0].imshow(smiles_to_image(tsmiles))
            set_image_caption(self._axs[r, 0], tsmiles, size=8)

            result = self._results[r]
            psmiles_list, recon_result = result

            # TODO caption iteration/max iterations
            last_step = self._step >= len(psmiles_list)

            psmiles = psmiles_list[min(self._step, len(psmiles_list) - 1)]
            self._axs[r, 1].imshow(smiles_to_image(psmiles))
            set_image_caption(self._axs[r, 1], psmiles, size=8)

            if last_step:  # If last step, show the result
                self._axs[r, 2] \
                    .text(0.5, 0.5, recon_result.name, transform=self._axs[r, 2].transAxes, ha='center', va='center')

            # Plot Tanimoto similarities
            self._axs[r, 3].set_ylim(0, 1)
            scores = self._scores[r]
            num_steps = len(scores)
            x = list(range(num_steps))
            y = [0] * num_steps
            self._axs[r, 3].set_xticks(np.arange(0, num_steps, 1))
            for step in range(min(self._step + 1, num_steps)):
                if step < num_steps:
                    y[step] = self._scores[r][step]
            self._axs[r, 3].scatter(x, y, color='blue')

    def _on_key_press(self, event):
        if event.key == 'enter':
            # plt.clf()
            self._step += 1
            if self._step > self._max_step:
                self._sample_tsmiles()
                self._step = 0
            self._redraw_canvas()
            plt.draw()

    def start(self):
        self._sample_tsmiles()
        self._redraw_canvas()

        plt.tight_layout()
        self._fig.canvas.mpl_connect('key_press_event', lambda event: self._on_key_press(event))
        plt.show()


def _main():
    context = parse_runtime_context_from_cmdline()
    ReconstructView(context).start()


if __name__ == "__main__":
    _main()
