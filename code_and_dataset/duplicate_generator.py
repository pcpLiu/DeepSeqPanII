import os
import sys

def generate_duplicate(i, template, folder):
    """Generate duplicate experiment folder
    """
    out_dir = '{}/dup_{}'.format(folder, i)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    template = open(template, 'r').read()
    out_str = template.replace('#WORKING_DIR#', out_dir)

    out_config_path = '{}/config.json'.format(out_dir)
    out_file = open(out_config_path, 'w')
    out_file.write(out_str)
    out_file.close()

    print('Create duplicate:', out_dir)


def main():
    for i in range(int(sys.argv[1])):
        generate_duplicate(i, sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()
