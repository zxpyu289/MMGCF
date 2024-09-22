from rdkit import DataStructs
from torch.nn import functional as F

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig

class Fingerprint:
    def __init__(self, fingerprint, fp_length):
        self.fp = fingerprint
        self.fp_len = fp_length

    def is_valid(self):
        return self.fingerprint is None

    def numpy(self):
        np_ = np.zeros((1,))
        ConvertToNumpyArray(self.fp, np_)
        return np_

    def tensor(self):
        return torch.as_tensor(self.numpy())


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)


def mfp(molecule, fp_len, fp_rad, bitInfo=None):
    m = molecule
    if isinstance(molecule, str):
        molecule = mol_from_smiles(molecule)

    if molecule is None:
        print(m)

    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, fp_rad, fp_len, bitInfo=bitInfo)
    return Fingerprint(fp, fp_len)

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def cosine_similarity(encoding_a, encoding_b):
    return F.cosine_similarity(encoding_a, encoding_b).item()



def get_similarity(name, model, original_molecule, original_smiles,fp_len=None, fp_rad=None):


    if name == "neural_encoding":
        similarity = lambda x, y: cosine_similarity(x, y)

        make_encoding = lambda x: model(x, x.ndata['node'].float(), x.edata['edge'].long(), x.ndata['smask'].unsqueeze(dim=1).float())[2]
        original_encoding = make_encoding(original_molecule)

    elif name == "combined":
        similarity = lambda x, y: 0.5 * cosine_similarity(x[0], y[0]) + 0.5 * tanimoto_similarity(x[1], y[1])

        # make_encoding = lambda x: (model(x.x, x.edge_index)[1][1], mfp(x.smiles, fp_len, fp_rad).fp)
        make_encoding = lambda x, smiles: (model(x, x.ndata['node'].float(), x.edata['edge'].long(), x.ndata['smask'].unsqueeze(dim=1).float())[2], mfp(smiles, fp_len, fp_rad).fp)
        original_encoding = make_encoding(original_molecule, original_smiles)

    return similarity, make_encoding, original_encoding
