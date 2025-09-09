from eval.stats3 import eval_graph_list
from fcd_torch import FCD
import pickle
import torch
import random
import numpy as np
import pandas as pd
import json
import networkx as nx

import re
from rdkit import Chem, RDLogger, rdBase
import rdkit.Chem.QED as QED
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE,
                4: Chem.rdchem.BondType.AROMATIC}
AN_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def fraction_unique(gen, k=None, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            raise ValueError(f"Can't compute unique@{k} gen contains only {len(gen)} molecules")
        gen = gen[:k]
    canonic = set(map(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)

def remove_invalid(gen, canonize=True):
    """
    Removes invalid molecules from the dataset
    """

    # if not canonize:
    #     mols = get_mol(gen)
    #     return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in map(canonic_smiles, gen) if x is not None]
#
# def fraction_valid(gen):
#     """
#     Computes a number of valid molecules
#     Parameters:
#         gen: list of SMILES
#     """
#     gen = [mol for mol in map(get_mol, gen)]
#     return 1 - gen.count(None) / len(gen)

def novelty(gen, train):
    gen_smiles = []
    for smiles in gen:
        gen_smiles.append(canonic_smiles(smiles))
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def mol_metric(gen_mols, dataset, num_no_correct, test_metrics=False):
    '''
    Args:
        - graphs(list of torch_geometric.Data)
        - train_smiles (list of smiles from the training set)
    Return:
        - Dict with key valid, unique, novel nspdk
    '''
    n = len(gen_mols)
    metrics = {}
    rdBase.DisableLog('rdApp.*')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen_smiles = mols_to_smiles(gen_mols)

    metrics['valid'] = num_no_correct
    gen_valid = remove_invalid(gen_smiles)
    if len(gen_valid) > 0:
        metrics['unique'] = fraction_unique(gen_valid, k=None, check_validity=True)
    else:
        metrics['unique'] = 0
    if test_metrics:
        if dataset == 'qm9_conditional' or dataset == 'qm9_cc' or dataset== 'qm9_dg':
            dataset = 'qm9'
        train_smiles, test_smiles = load_smiles(dataset=dataset)
        if len(gen_valid) > 0:
            metrics['novel'] = novelty(gen_valid, train_smiles)
        else:
            metrics['novel'] = 0

        with open(f'./data/{dataset}/raw/test_nx_{dataset}.pkl', 'rb') as f:
            test_graph_list = pickle.load(f)
            random.Random(42).shuffle(test_graph_list)

        if len(gen_mols) > 0:
            metrics['nspdk'] = eval_graph_list(test_graph_list[:n], mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        else:
            metrics['nspdk'] = 1

        metrics['fcd'] = FCD(n_jobs=0, device=device)(ref=test_smiles, gen=gen_smiles)
        metrics['valid_with_corr'] = len(gen_valid)
    return metrics

# Code adapted from GDSS harryjo/GDSS/utils/smile_to_graph.py (only cosmetic changes)

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def load_smiles(dataset='qm9'):
    if dataset == 'qm9':
        col = 'SMILES1'
    elif dataset == 'zinc':
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')

    df = pd.read_csv(f'./data/{dataset}/raw/smiles_{dataset}.csv')

    with open(f'./data/{dataset}/raw/test_idx_{dataset}.json') as f:
        test_idx = json.load(f)

    if dataset == 'qm9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    train_idx = [i for i in range(len(df)) if i not in test_idx]
    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])


def gen_mol(x, adj, mask, dataset, largest_connected_comp=True):
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if dataset == 'qm9':
        atomic_num_list = [6, 7, 8, 9]
    elif dataset == 'qm9H':
        atomic_num_list = [6, 7, 8, 9]
    elif dataset == 'qm9_conditional':
        atomic_num_list = [1, 6, 7, 8, 9, 0]
    elif dataset in ['qm9_cc', 'qm9_dg', 'qm9H']:
        atomic_num_list = [1, 6, 7, 8, 9, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    # mols_wo_correction = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    # mols_wo_correction = [mol for mol in mols_wo_correction if mol is not None]
    mols, num_no_correct = [], 0
    for x, a, m in zip(x, adj, mask):
        a = a[m]
        a = a[..., m]
        mol = construct_mol(x[m], a, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct:
            num_no_correct += 1
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        mols.append(vcmol)

    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def gen_mol_no_corr(x, adj, dataset):
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()

    if dataset == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
    elif dataset in ['qm9_cc', 'qm9_dg', 'qm9H']:
        atomic_num_list = [1, 6, 7, 8, 9, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    # mols_wo_correction = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    # mols_wo_correction = [mol for mol in mols_wo_correction if mol is not None]
    mols = []
    for x_elem, adj_elem in zip(x, adj):
        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        mols.append(mol)
    mols = [mol for mol in mols if mol is not None]
    return mols


def construct_mol(atoms, adj, atomic_num_list):  # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()
    #atoms = np.argmax(x, axis=1)
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # adj = np.argmax(adj, axis=0)  # 9, 9
    # adj = adj[atoms_exist, :][:, atoms_exist]
    # adj[adj == 3] = -1
    # adj += 1  # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
            #    atomic_num=atom.GetAtomicNum(),
            #    formal_charge=atom.GetFormalCharge(),
            #    chiral_tag=atom.GetChiralTag(),
            #    hybridization=atom.GetHybridization(),
            #    num_explicit_hs=atom.GetNumExplicitHs(),
            #    is_aromatic=atom.GetIsAromatic())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
            #    bond_type=bond.GetBondType())

        nx_graphs.append(G)
    return nx_graphs



def get_molecular_properties(mols):
    values = torch.zeros(0, 3)
    for mol in mols:
        try:
            qed = QED.qed(mol) #- min_qed
            # qed = qed / range_qed
        except:
            qed = 0
        weight = Descriptors.MolWt(mol) #- min_weight
        # weight = weight / range_weight

        logP = Chem.Crippen.MolLogP(mol) #- min_logP
        # logP = logP / range_logP
        values = torch.cat((values, torch.tensor([weight, logP, qed]).unsqueeze(0)), dim=0)
    return values

