import os
import math
import numbers
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
#############################################################################################
#
# Write result
#
#############################################################################################

def weeekly_result_writer(result_dict, config):
    """Write prediction result as an additional column
    out [weekly_result.txt]
    """
    out_file_path = os.path.join(config.working_dir, 'weekly_result.txt')
    out_file = open(out_file_path, 'w')

    with open(config.test_file) as in_file:
        for line_num, line in enumerate(in_file):
            info = line.strip('\n').split('\t')
            if line_num == 0:
                # title
                out_str = '{}\t{}\t{}\n'.format(
                    '\t'.join(info[:7]),
                    'our_method_ic50',
                    '\t'.join(info[7:])
                )
                out_file.write(out_str)
            else:
                iedb_id = info[1]
                alleles = info[2].lstrip('HLA-')
                measure_type = info[4]
                peptide = info[5]

                hla_a = alleles
                hla_b = alleles
                if '/' in alleles:
                    hla_a = alleles.split('/')[0]
                    hla_b = alleles.split('/')[1]

                uid = '{iedb_id}-{hla_a}-{hla_b}-{peptide}-{measure_type}'.format(
                    iedb_id=iedb_id,
                    hla_a=hla_a,
                    hla_b=hla_b,
                    peptide=peptide,
                    measure_type=measure_type,
                )

                if uid not in result_dict:
                    value = '-'
                else:
                    value = math.pow(50000, 1 - result_dict[uid])
                out_str = '{}\t{}\t{}\n'.format(
                    '\t'.join(info[:7]),
                    value,
                    '\t'.join(info[7:])
                )
                out_file.write(out_str)

    return out_file_path

#############################################################################################
#
# Write metrics
#
#############################################################################################

METHOD_LIST = [
    'our_method_ic50',
    'NN-align',
    'NetMHCIIpan-3.1',
    'Comblib matrices',
    'SMM-align',
    'Tepitope (Sturniolo)',
    'Consensus IEDB method',
]


def get_srcc(real, pred, measure_type):
    """
    """
    # all pred are ic50, neg them to get real correlation
    pred = [-x for x in pred]

    # if real also ic50, neg them
    if measure_type == 'ic50':
        real = [-x for x in real]

    return spearmanr(pred, real)[0]


def get_auc(real, pred, measure_type):
    """
    """
    # all pred are ic50, neg them to get real correlation
    pred = [-x for x in pred]

    # convert real to binary labels according to measure type
    real_binary = real
    if measure_type == 'ic50':
        real_binary = [1 if x < 500 else 0 for x in real]

    try:
        return roc_auc_score(real_binary, pred)
    except:
        return '-'


def get_weekly_result_info_dict(result_file, ignore_pep_length=False, ignore_iedb_id_and_length=False):
    """Reading from [weekly_result.txt], get info for each record dict
    Return dict format:
    {
        'iedb_id-allele-pep_len-measure_type': {
            'date': x,
            'pep_length': x,
            'iedb_id':x,
            'full_allele':x,
            'measure_type': x,
            'method_values': {
                'method_name': [x,xx,x],
                ....
            },
            'label_values': [x,x,x,x]
        }
    }
    """
    result_info = {}
    with open(result_file) as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            info = line.strip('\n').split('\t')
            date = info[0]
            iedb_id = info[1]
            full_allele = info[2]
            measure_type = info[4]
            measure_value = float(info[6])

              # ignor non human
            if 'HLA' not in full_allele:
                continue

            if ignore_pep_length:
                record_id = '{}-{}-{}'.format(iedb_id, full_allele, measure_type)
            elif ignore_iedb_id_and_length:
                record_id = '{}-{}'.format(full_allele, measure_type)
            else:
                pep_len = len(info[5])
                record_id = '{}-{}-{}-{}'.format(iedb_id, full_allele, pep_len, measure_type)

            if record_id not in result_info:
                result_info[record_id] = {}
                result_info[record_id]['full_allele'] = full_allele
                result_info[record_id]['date'] = date
                result_info[record_id]['iedb_id'] = iedb_id
                result_info[record_id]['measure_type'] = measure_type
                result_info[record_id]['label_values'] = []
                result_info[record_id]['method_values'] = {}
                if not ignore_pep_length and not ignore_iedb_id_and_length:
                    result_info[record_id]['pep_length'] = pep_len

                for method in METHOD_LIST:
                    result_info[record_id]['method_values'][method] = []

            # fill real value
            result_info[record_id]['label_values'].append(measure_value)

            # fill prediction values, if no result, do not fill
            for method_index, method_name in enumerate(METHOD_LIST):
                col_index = method_index + 7
                val = info[col_index]
                try:
                    val = float(val)
                    result_info[record_id]['method_values'][method_name].append(val)
                except:
                    pass

    return result_info


def write_metrics_file(result_file, config):
    """Reading [weekly_result.txt], write to [weekly_result_METRICS.txt]
    by each IEDB record
    """
    METRIC_PRECISION_DIGIT = 2

    out_file_path = os.path.join(config.working_dir, 'weekly_result_METRICS.txt')
    out_file = open(out_file_path, 'w')

    title = 'Date\tIEDB reference\tAllele\tPeptide length\tcount\tMeasurement type'

    for method_name in METHOD_LIST:
        title += '\t{method_name}_auc\t{method_name}_srcc'.format(method_name=method_name)
    out_file.write(title + '\n')

    # use to get max value for each record
    metric_max_info = {}
    for method_name in METHOD_LIST:
        metric_max_info[method_name] = [0, 0]

    result_info = get_weekly_result_info_dict(result_file)
    for record, info in result_info.items():
        date = info['date']
        iedb_id = info['iedb_id']
        pep_length = info['pep_length']
        measure_type = info['measure_type']
        allele = info['full_allele']
        label_values = info['label_values']
        count = len(label_values)

        out_str = '{}\t{}\t{}\t{}\t{}\t{}'.format(
            date, iedb_id, allele, pep_length, count, measure_type
        )

        max_srcc = -1000000
        max_auc = -1000000
        srcc_list = []
        auc_list = []
        for method_name in METHOD_LIST:
            pred_vals = info['method_values'][method_name]
            if len(pred_vals) != count:
                srcc = '-'
                auc = '-'
            else:
                srcc = get_srcc(label_values, pred_vals, measure_type)
                auc = get_auc(label_values, pred_vals, measure_type)
                if isinstance(srcc, numbers.Number):
                    max_srcc = max(srcc, max_srcc)
                if isinstance(auc, numbers.Number):
                    max_auc = max(auc, max_auc)

            srcc_list.append(srcc)
            auc_list.append(auc)
            out_str += '\t{}\t{}'.format(auc, srcc)

        # update max auc, srcc count in this record
        for i, (srcc, auc) in enumerate(zip(srcc_list, auc_list)):
            if auc == max_auc:
                metric_max_info[METHOD_LIST[i]][0] += 1
            if srcc == max_srcc:
                metric_max_info[METHOD_LIST[i]][1] += 1

        # write
        out_file.write(out_str + '\n')

    # write max win count
    out_str = '\t'.join(['-'] * 6)  # offset
    for method_name in METHOD_LIST:
        out_str += '\t{}\t{}'.format(metric_max_info[method_name][0], metric_max_info[method_name][1])
    out_file.write(out_str + '\n')

    return out_file_path


def write_metrics_file_IGNORE_IEDBID_AND_LENGTH(result_file, config):
    """Reading [weekly_result.txt], write to [weekly_result_METRICS_IGNORE_IEDB_ID_AND_LENGTH.txt]
    by each IEDB record, but ignore peptide length and iedb id for a same allele
    """
    METRIC_PRECISION_DIGIT = 2

    out_file_path = os.path.join(config.working_dir, 'weekly_result_METRICS_IGNORE_IEDB_ID_AND_LENGTH.txt')
    out_file = open(out_file_path, 'w')

    title = 'Date\tAllele\tcount\tMeasurement type'
    for method_name in METHOD_LIST:
        title += '\t{method_name}_auc\t{method_name}_srcc'.format(method_name=method_name)
    out_file.write(title + '\n')

    # use to get max value for each record
    metric_max_info = {}
    for method_name in METHOD_LIST:
        metric_max_info[method_name] = [0, 0]

    result_info = get_weekly_result_info_dict(result_file, ignore_iedb_id_and_length=True)
    for record, info in result_info.items():
        date = info['date']
        iedb_id = info['iedb_id']
        measure_type = info['measure_type']
        allele = info['full_allele']
        label_values = info['label_values']
        count = len(label_values)

        out_str = '{}\t{}\t{}\t{}'.format(
            date, allele, count, measure_type
        )

        max_srcc = -1000000
        max_auc = -1000000
        srcc_list = []
        auc_list = []
        for method_name in METHOD_LIST:
            pred_vals = info['method_values'][method_name]
            if len(pred_vals) != count:
                srcc = '-'
                auc = '-'
            else:
                srcc = get_srcc(label_values, pred_vals, measure_type)
                auc = get_auc(label_values, pred_vals, measure_type)
                if isinstance(srcc, numbers.Number):
                    max_srcc = max(srcc, max_srcc)
                if isinstance(auc, numbers.Number):
                    max_auc = max(auc, max_auc)

            srcc_list.append(srcc)
            auc_list.append(auc)
            out_str += '\t{}\t{}'.format(auc, srcc)

        # update max auc, srcc count in this record
        for i, (srcc, auc) in enumerate(zip(srcc_list, auc_list)):
            if auc == max_auc:
                metric_max_info[METHOD_LIST[i]][0] += 1
            if srcc == max_srcc:
                metric_max_info[METHOD_LIST[i]][1] += 1

        # write
        out_file.write(out_str + '\n')

    # write max win count
    out_str = '\t'.join(['-'] * 4)  # offset
    for method_name in METHOD_LIST:
        out_str += '\t{}\t{}'.format(metric_max_info[method_name][0], metric_max_info[method_name][1])
    out_file.write(out_str + '\n')

    return out_file_path


#############################################################################################
#
# Write binding core
#
#############################################################################################

def attn_weight_core_fetch(attn_weight, peptide):
    """Accoding to attn_weight to fetch max 9 position
    Note: we don consider padded sequenc after valid
    """
    max_weight = -float('Inf')
    core_bind = ''
    for start_i in range(0, len(peptide) - 9 + 1):
        sum_weight = sum(attn_weight[start_i: start_i + 9])
        if sum_weight > max_weight:
            max_weight = sum_weight
            core_bind = peptide[start_i: start_i + 9]
    return core_bind


def write_binding_core_results(attn_weight_dict, config):
    """According to attention weight, write out peptide binding core.
    This version: the subsequence with max sum weight
    """
    out_file_path = os.path.join(config.working_dir, 'binding_core_results.txt')
    out_file = open(out_file_path, 'w')

    with open(config.bind_core_file) as in_file:
        for line_num, line in enumerate(in_file):
            info = line.strip('\n').split('\t')
            if line_num == 0:
                # title
                out_str = 'PDB\thla_a\thla_b\tcore_pdb\tcore_pred\tweight_list\n'
                out_file.write(out_str)
            else:
                pbd_id = info[0]
                alleles = info[1]
                peptide = info[2]
                bind_core = info[3]

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

                # get core bind
                attn_weight = attn_weight_dict[uid]
                pred_bind_core = attn_weight_core_fetch(attn_weight, peptide)

                out_str = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    pbd_id,
                    hla_a,
                    hla_b,
                    peptide,
                    bind_core,
                    pred_bind_core,
                    ','.join([str(x) for x in attn_weight])
                )
                out_file.write(out_str)

    return out_file_path
