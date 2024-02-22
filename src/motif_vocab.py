import pandas as pd
from common import *


class MotifVocab(pd.DataFrame):
    """ A class representing the Motif vocabulary dataframe. """

    def __iter__(self):
        for i, row in self.iterrows():
            yield row['motif']

    @staticmethod
    def load():
        df = pd.read_csv(VOCAB_PATH)
        return MotifVocab(df[df['is_motif']]['motif'])
