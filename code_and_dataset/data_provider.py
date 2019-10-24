import math
import os
import random

import torch

from logger import log_to_file

############################################################################
# Data provider
############################################################################


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class DataProvider:
    def __init__(self, sequence_encode_func, data_file, test_file, bind_core_file, batch_size, max_len_hla_A=274, max_len_hla_B=291, max_len_pep=37,
                 validation_ratio=0.2, shuffle=True, LOMO=False, LOMO_allele=None,
                 shuffle_before_epoch_enable=False,):
        self.batch_size = batch_size
        self.data_file = data_file
        self.test_file = test_file
        self.bind_core_file = bind_core_file
        self.sequence_encode_func = sequence_encode_func
        self.shuffle = shuffle
        self.validation_ratio = validation_ratio
        self.LOMO = LOMO
        self.LOMO_allele = LOMO_allele
        self.shuffle_before_epoch_enable = shuffle_before_epoch_enable

        self.max_len_hla_A = max_len_hla_A
        self.max_len_hla_B = max_len_hla_B
        self.max_len_pep = max_len_pep

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0
        self.batch_index_bind_core = 0

        # cache
        self.pep_encode_dict = {}
        self.hla_encode_dict = {}

        # bind core
        self.bind_core_samples = []
        self.read_bind_core_file()

        self.hla_sequence_A = {}
        self.hla_sequence_B = {}
        self.read_hla_sequences()

        self.samples = []  # list of tuple (hla_a, hla_b, peptide, ic50)
        self.train_samples = []
        self.validation_samples = []
        self.read_training_data()

        if self.LOMO:
            # list of tupe (hla_a, hla_b, peptide, ic50)
            self.lomo_test_samples = []
            self.split_train_and_val_and_test()
        else:
            # list of tupe (hla_a, hla_b, peptide, uid)
            self.split_train_and_val()
            self.weekly_samples = []
            self.read_weekly_data()

    def train_steps(self):
        return math.ceil(len(self.train_samples) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)

    def LOMO_test_steps(self):
        return math.ceil(len(self.lomo_test_samples) / self.batch_size)

    def bind_core_steps(self):
        return math.ceil(len(self.bind_core_samples) / self.batch_size)

    def read_hla_sequences(self):
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

        read('CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt', self.hla_sequence_A)
        read('CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt', self.hla_sequence_B)

    def read_weekly_data(self):
        """Read weekly data, most like [all_weekly.txt]
        """
        with open(self.test_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                iedb_id = info[1]
                alleles = info[2].lstrip('HLA-')
                measure_type = info[4]
                peptide = info[5]

                hla_a = alleles
                hla_b = alleles
                if '/' in alleles:
                    hla_a = alleles.split('/')[0]
                    hla_b = alleles.split('/')[1]

                # filter out alleles we dont have sequences
                if hla_a not in self.hla_sequence_A or hla_b not in self.hla_sequence_B:
                    continue

                uid = '{iedb_id}-{hla_a}-{hla_b}-{peptide}-{measure_type}'.format(
                    iedb_id=iedb_id,
                    hla_a=hla_a,
                    hla_b=hla_b,
                    peptide=peptide,
                    measure_type=measure_type,
                )

                self.weekly_samples.append((hla_a, hla_b, peptide, uid))

    def read_bind_core_file(self):
        """Read target bind core file, format like [binding_core.txt]
        """
        with open(self.bind_core_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                pbd_id = info[0]
                alleles = info[1]
                peptide = info[2]

                if 'H-2' in alleles:
                    continue

                hla_a = alleles
                hla_b = alleles
                if '-' in alleles:
                    hla_a = alleles.split('-')[0]
                    hla_b = alleles.split('-')[1]

                uid = '{pbd_id}-{hla_a}-{hla_b}-{peptide}'.format(
                    pbd_id=pbd_id,
                    hla_a=hla_a,
                    hla_b=hla_b,
                    peptide=peptide,
                )

                self.bind_core_samples.append((hla_a, hla_b, peptide, uid))

    def read_training_data(self):
        """Read training data, most likely [BD_2013_DATAPROVIDER_READY.txt]
        """
        with open(self.data_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                if line[0] == "#":
                    continue

                info = line.strip('\n').split('\t')

                # get two peptide
                hla_a = info[0]
                hla_b = info[0]
                if '-' in info[0]:
                    hla_a = info[0].split('-')[0]
                    hla_b = info[0].split('-')[1]

                # filter out alleles we dont have sequences
                if hla_a not in self.hla_sequence_A or hla_b not in self.hla_sequence_B:
                    continue

                peptide = info[1]
                # filter out peptide we dont want to see
                if len(peptide) > self.max_len_pep:
                    continue

                ic50 = float(info[-1])

                self.samples.append((hla_a, hla_b, peptide, ic50))

        if self.shuffle:
            random.shuffle(self.samples)

    def split_train_and_val_and_test(self):
        """Split training and validation.
        We split validation by select self.validation_ratio from each allele
        """
        self.train_samples = []
        self.validation_samples = []

        # filter out testing allele samples
        test_a, test_b = self.LOMO_allele.split('-')
        self.lomo_test_samples = list(
            filter(lambda x: x[0] == test_a and x[1] == test_b, self.samples))

        train_set = list(
            filter(lambda x: x[0] != test_a or x[1] != test_b, self.samples))
        train_alleles = set(map(lambda x: (x[0], x[1]), train_set))

        def split_train_val_by_allele(allele):
            this_allele_samples = list(
                filter(lambda x: x[0] == allele[0] and x[1] == allele[1], train_set))

            # not enough, directly add to training
            if len(this_allele_samples) <= 1:
                self.train_samples.extend(this_allele_samples)
                return

            # shuffle
            random.shuffle(this_allele_samples)

            val_count = max(1, math.floor(
                len(this_allele_samples) * self.validation_ratio))
            self.validation_samples.extend(this_allele_samples[:val_count])
            self.train_samples.extend(this_allele_samples[val_count:])

        for allele in train_alleles:
            split_train_val_by_allele(allele)

        random.shuffle(self.train_samples)

    def split_train_and_val(self):
        """Split training and validation by val ratio
        """
        self.train_samples = []
        self.validation_samples = []

        train_alleles = set(map(lambda x: (x[0], x[1]), self.samples))

        def split_train_val_by_allele(allele):
            this_allele_samples = list(
                filter(lambda x: x[0] == allele[0] and x[1] == allele[1], self.samples))

            # not enough, directly add to training
            if len(this_allele_samples) <= 1:
                self.train_samples.extend(this_allele_samples)
                return

            # shuffle
            random.shuffle(this_allele_samples)

            val_count = max(1, math.floor(
                len(this_allele_samples) * self.validation_ratio))
            self.validation_samples.extend(this_allele_samples[:val_count])
            self.train_samples.extend(this_allele_samples[val_count:])

        for allele in train_alleles:
            split_train_val_by_allele(allele)

        random.shuffle(self.train_samples)

    def batch_train(self):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples)
        self.batch_index_train += 1
        return data

    def batch_val(self):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples)
        self.batch_index_val += 1
        return data

    def batch_test(self):
        """A batch of test data
        """
        if self.LOMO:
            data = self.batch(self.batch_index_test,
                              self.lomo_test_samples, LOMO_testing=True)
            self.batch_index_test += 1
            return data
        else:
            data = self.batch(self.batch_index_test,
                              self.weekly_samples, testing=True)
            self.batch_index_test += 1
            return data

    def batch_bind_core(self):
        """A batch of bind core samples
        """
        data = self.batch(self.batch_index_bind_core,
                          self.bind_core_samples, testing=True)
        self.batch_index_bind_core += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

        if self.shuffle_before_epoch_enable:
            log_to_file("\t"*4, "Re-split train & val")
            self.split_train_and_val()
        else:
            log_to_file("\t"*4, "Not enable Re-split train & val")

    def batch(self, batch_index, sample_set, testing=False, LOMO_testing=False):
        """Get a batch of samples
        """
        hla_a_tensors = []
        hla_a_mask = []
        hla_a_length = []

        hla_b_tensors = []
        hla_b_mask = []
        hla_b_length = []

        pep_tensors = []
        pep_mask = []
        pep_length = []

        ic50_list = []

        # for testing
        uid_list = []

        # for LOMO testing
        pep_seq_list = []

        def encode_sample(sample):
            hla_a_allele = sample[0]
            hla_b_allele = sample[1]
            pep = sample[2]

            if not testing:
                ic50 = sample[3]
            else:
                uid = sample[3]

            if hla_a_allele not in self.hla_encode_dict:
                hla_a_tensor, mask, last_valid_index = self.sequence_encode_func(
                    self.hla_sequence_A[hla_a_allele], self.max_len_hla_A)
                self.hla_encode_dict[hla_a_allele] = (
                    hla_a_tensor, mask, last_valid_index)
            hla_a_tensors.append(self.hla_encode_dict[hla_a_allele][0])
            hla_a_mask.append(self.hla_encode_dict[hla_a_allele][1])
            hla_a_length.append(self.hla_encode_dict[hla_a_allele][2])

            if hla_b_allele not in self.hla_encode_dict:
                hla_b_tensor, mask, last_valid_index = self.sequence_encode_func(
                    self.hla_sequence_B[hla_b_allele], self.max_len_hla_B)
                self.hla_encode_dict[hla_b_allele] = (
                    hla_b_tensor, mask, last_valid_index)
            hla_b_tensors.append(self.hla_encode_dict[hla_b_allele][0])
            hla_b_mask.append(self.hla_encode_dict[hla_b_allele][1])
            hla_b_length.append(self.hla_encode_dict[hla_b_allele][2])

            if pep not in self.pep_encode_dict:
                pep_tensor, mask, last_valid_index = self.sequence_encode_func(
                    pep, self.max_len_pep)
                self.pep_encode_dict[pep] = (
                    pep_tensor, mask, last_valid_index)
            pep_tensors.append(self.pep_encode_dict[pep][0])
            pep_mask.append(self.pep_encode_dict[pep][1])
            pep_length.append(self.pep_encode_dict[pep][2])

            if not testing:
                ic50_list.append(ic50)
            else:
                uid_list.append(uid)

            if LOMO_testing:
                pep_seq_list.append(pep)

        start_i = batch_index * self.batch_size
        end_i = start_i + self.batch_size
        for sample in sample_set[start_i: end_i]:
            encode_sample(sample)

        # in case last batch does not have enough samples, random get from previous samples
        if len(hla_a_tensors) < self.batch_size:
            if len(sample_set) < self.batch_size:
                for _ in range(self.batch_size - len(hla_a_tensors)):
                    encode_sample(random.choice(sample_set))
            else:
                for i in random.sample(range(start_i), self.batch_size - len(hla_a_tensors)):
                    encode_sample(sample_set[i])

        if not testing:
            if LOMO_testing:
                return (
                    torch.stack(hla_a_tensors, dim=0),
                    torch.stack(hla_a_mask, dim=0),
                    torch.tensor(hla_a_length),

                    torch.stack(hla_b_tensors, dim=0),
                    torch.stack(hla_b_mask, dim=0),
                    torch.tensor(hla_b_length),

                    torch.stack(pep_tensors, dim=0),
                    torch.stack(pep_mask, dim=0),
                    torch.tensor(pep_length),

                    torch.tensor(ic50_list),
                    pep_seq_list,
                )
            else:
                return (
                    torch.stack(hla_a_tensors, dim=0),
                    torch.stack(hla_a_mask, dim=0),
                    torch.tensor(hla_a_length),

                    torch.stack(hla_b_tensors, dim=0),
                    torch.stack(hla_b_mask, dim=0),
                    torch.tensor(hla_b_length),

                    torch.stack(pep_tensors, dim=0),
                    torch.stack(pep_mask, dim=0),
                    torch.tensor(pep_length),

                    torch.tensor(ic50_list),
                )
        else:
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.tensor(hla_a_length),

                torch.stack(hla_b_tensors, dim=0),
                torch.stack(hla_b_mask, dim=0),
                torch.tensor(hla_b_length),

                torch.stack(pep_tensors, dim=0),
                torch.stack(pep_mask, dim=0),
                torch.tensor(pep_length),

                uid_list,
            )
