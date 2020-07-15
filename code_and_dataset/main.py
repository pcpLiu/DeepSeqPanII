import os
import torch
import sys
from seq_encoding import one_hot_PLUS_blosum_encode
from config_parser import Config
from model import Model


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def read_hla_sequences():
    """Read hla sequences from [CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt]
    and [CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt]
    """
    def read(f, d):
        file_path = os.path.join(BASE_DIR, 'dataset', f)
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                d[info[0]] = info[1]

    hla_sequence_A = {}
    hla_sequence_B = {}
    read('CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt', hla_sequence_A)
    read('CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt', hla_sequence_B)
    return hla_sequence_A, hla_sequence_B


def run(model_path, hla_a, hla_b, peptide):
    """Get ic50
    """
    # load model
    config = Config("config_main.json")
    config.device = 'cpu'
    state_dict = torch.load(os.path.join(BASE_DIR, model_path))
    model = Model(config)
    model.load_state_dict(state_dict)
    model.eval()

    peptide_encoded, pep_mask, pep_len = one_hot_PLUS_blosum_encode(peptide, config.max_len_pep)
    hla_sequence_A, hla_sequence_B = read_hla_sequences()
    hla_a_seq = hla_sequence_A[hla_a]
    hla_b_seq = hla_sequence_B[hla_b]
    hla_a_encoded, hla_a_mask, hla_a_len = one_hot_PLUS_blosum_encode(hla_a_seq, config.max_len_hla_A)
    hla_b_encoded, hla_b_mask, hla_b_len = one_hot_PLUS_blosum_encode(hla_b_seq, config.max_len_hla_B)

    pred_ic50, _ = model(
        torch.stack([hla_a_encoded], dim=0),
        torch.stack([hla_a_mask], dim=0),
        torch.tensor([hla_a_len]),

        torch.stack([hla_b_encoded], dim=0),
        torch.stack([hla_b_mask], dim=0),
        torch.tensor([hla_b_len]),

        torch.stack([peptide_encoded], dim=0),
        torch.stack([pep_mask], dim=0),
        torch.tensor([pep_len]),
    )
    print("IC 50: ", pred_ic50.item())

if __name__ == '__main__':
    run(
        model_path=sys.argv[1],
        hla_a=sys.argv[2],
        hla_b=sys.argv[3],
        peptide=sys.argv[4],
    )
