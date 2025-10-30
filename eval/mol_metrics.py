from eval.stats3 import eval_graph_list
from fcd_torch import FCD
import pickle
import torch
import random
import numpy as np
import pandas as pd
import json
import networkx as nx

from multiprocessing import Pool
from collections import Counter
from functools import partial
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.six import iteritems
import scipy
import math
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine as cos_distance
import os.path as op

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
        test_smiles = canonicalize_smiles(test_smiles)
        train_smiles = canonicalize_smiles(train_smiles)

        if len(gen_valid) > 0:
            metrics['novel'] = novelty(gen_valid, train_smiles)
        else:
            metrics['novel'] = 0

        with open(f'./data/{dataset}/raw/test_nx_{dataset}.pkl', 'rb') as f:
            test_graph_list = pickle.load(f)
            random.Random(42).shuffle(test_graph_list)

        if len(gen_mols) > 0:
            metrics['nspdk'] = eval_graph_list(test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        else:
            metrics['nspdk'] = 1

        metrics['fcd'] = FCD(n_jobs=0, device=device)(ref=test_smiles, gen=gen_smiles)

        kwargs = {'n_jobs': 1, 'device': device, 'batch_size': len(gen_smiles)}
        ptest = compute_intermediate_statistics(test_smiles, n_jobs=1,
                                                device=device,
                                                batch_size=len(gen_smiles),
                                                pool=1)
        metrics['SNN/Test'] = SNNMetric(**kwargs)(gen=gen_mols, pref=ptest['SNN'])
        metrics['Frag/Test'] = FragMetric(**kwargs)(gen=gen_mols, pref=ptest['Frag'])
        metrics['Scaf/Test'] = ScafMetric(**kwargs)(gen=gen_mols, pref=ptest['Scaf'])
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
    elif dataset == 'qm9_conditional':
        atomic_num_list = [1, 6, 7, 8, 9, 0]
    elif dataset == ['qm9_cc','qm9_dg', 'qm9H']:
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
                if queue[0][1] == 12:
                    t = 1
                else:
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

def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    # statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED),
                       ('weight', weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics

class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError

class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(
            pref['values'], pgen['values']
        )

class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])

class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)

def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles


def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: # type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


_fscores = None

def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    print(name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateSAScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2  # <- 2 is the *radius* of the circular fingerprint
    )
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (0. - sizePenalty - stereoPenalty -
              spiroPenalty - bridgePenalty - macrocyclePenalty)

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return calculateSAScore(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)