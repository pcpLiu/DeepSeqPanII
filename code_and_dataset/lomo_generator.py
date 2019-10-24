import os
import sys

TEMPLATE = 'config_template_LOMO_BD2016.json'

def get_allele_list_BD2016():
    """Get alleel list from [../../datBD2016_allele_info_stat.xt]
    """
    allele_list = []

    with open('../../dataset/BD2016_allele_info_stat.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            info = line.strip('\n').split('\t')
            if int(info[1]) >= 10:
                allele_list.append(info[0])

    return allele_list


def generate_data_folder():
    allele_list = get_allele_list_BD2016()
    out_SH_file = open('run_LOMO_2016.sh', 'w')
    for allele in allele_list:
        out_dir = '{}/{}'.format('BD2016_LOMO', allele.replace('*', ''))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        config_content = open(TEMPLATE, 'r').read()
        config_content = config_content.replace('#TEST_ALLELE#', allele)
        config_content = config_content.replace('#WORKING_DIR#', out_dir)

        config_f_path = '{}/config.json'.format(out_dir)
        out_config_file = open(config_f_path, 'w')
        out_config_file.write(config_content)

        # sh command
        out_SH_file.write('python run.py {}\n'.format(config_f_path))

if __name__ == "__main__":
    generate_data_folder()
    pass
