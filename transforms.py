from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose
import rdkit.Chem.QED as QED
from rdkit.Chem import Descriptors
import torch

class Qm9Transform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data):
        data.x = data.x[..., :5]
        return data

class Qm9HTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data, mol: Chem.rdchem.Mol):
        data.x = data.x[..., :5]
        return data

class Qm9ConditionalTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        # self.min_qed =  0.10425393634515717
        # self.max_qed =  0.668797209018646
        # self.range_qed = self.max_qed - self.min_qed
        #
        # self.min_weight = 16.043
        # self.max_weight = 152.037
        # self.range_weight = self.max_weight - self.min_weight
        #
        # self.min_logP = -5.0039
        # self.max_logP = 3.7569
        # self.range_logP = self.max_logP - self.min_logP
        self.mean = torch.tensor([123.0129, 0.1307, 0.4617])
        self.std = torch.tensor([7.6222, 1.1716, 0.0769])

    def __call__(self, data: Data, mol: Chem.rdchem.Mol):

        #mol = Chem.MolFromSmiles(data.smiles, sanitize=True)

        try:
            qed = QED.qed(mol)# - self.min_qed
            # qed = qed / self.range_qed
            # data.qed = True
        except:
            # data.qed = False
            qed = self.mean[-1]
        weight = Descriptors.MolWt(mol)# - self.min_weight
        # weight = weight / self.range_weight

        logP = Chem.Crippen.MolLogP(mol) #- self.min_logP
        # logP = logP / self.range_logP

        data.x = data.x[..., :5]
        data.c = torch.tensor([weight, logP, qed]).unsqueeze(0)
        data.c = (data.c - self.mean)/self.std
        return data


class Qm9DiscreteGuidanceTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        # self.min_qed =  0.10425393634515717
        # self.max_qed =  0.668797209018646
        # self.range_qed = self.max_qed - self.min_qed
        #
        # self.min_weight = 16.043
        # self.max_weight = 152.037
        # self.range_weight = self.max_weight - self.min_weight
        #
        # self.min_logP = -5.0039
        # self.max_logP = 3.7569
        # self.range_logP = self.max_logP - self.min_logP
        self.mean = torch.tensor([123.0129, 0.1307, 0.4617])
        self.std = torch.tensor([7.6222, 1.1716, 0.0769])

    def __call__(self, data: Data, mol: Chem.rdchem.Mol):

        try:
            qed = QED.qed(mol)# - self.min_qed
            # qed = qed / self.range_qed
            # data.qed = True
        except:
            # data.qed = False
            qed = self.mean[-1]

        qed = self.discretize_continuous(qed, 0.1, 0.7, 0.05)


        weight = Descriptors.MolWt(mol)# - self.min_weight
        weight = self.discretize_continuous(weight, 15, 165, 15)

        logP = Chem.Crippen.MolLogP(mol)
        logP = self.discretize_continuous(logP, -5., 4, 1)

        data.x = data.x[..., :5]
        data.mol_weight = torch.tensor([weight, logP, qed]).unsqueeze(0)
        data.logP = logP
        data.qed = qed
        return data

    def discretize_continuous(self, value, start, stop, step):
        """
        Discretizes continuous values into categorical bins.

        Args:
            values (Tensor): Continuous input values to be discretized.
            start (float): Start of the range.
            stop (float): Stop of the range.
            step (float): Step size to define bin edges (like torch.linspace).

        Returns:
            Tensor: Discretized categorical values.
        """
        # Define bin edges using start, stop, and step
        bins = torch.arange(start, stop + step, step)  # Ensure last bin is included

        # Digitize values to get bin indices (torch.bucketize is a numerically stable alternative)
        categories = torch.bucketize(value, bins) - 1  # Subtract 1 so bins align with categories starting at 0

        # Ensure values below `start` go to the first category and above `stop` go to the last
        categories = torch.clamp(categories, min=0, max=len(bins) - 2)

        return categories
