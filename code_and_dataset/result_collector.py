import sys, os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def collect_result():
    """Collect result from [weekly_result_METRICS_IGNORE_LENGTH.txt] from dup
    """
    dup = int(sys.argv[2])
    bd = sys.argv[1]

    out_file = open('RESULT_COLLECTOR_{}.txt'.format(bd), 'w')

    for i in range(dup):
        result_file = os.path.join(BASE_DIR, '{}/dup_{}/weekly_result_METRICS_IGNORE_LENGTH.txt'.format(bd, i))
        with open(result_file, 'r') as f:
            for line in f:
                pass
            out_file.write(line)

def collect_result2():
    """Collect result from [weekly_result_METRICS_IGNORE_IEDB_ID_AND_LENGTH.txt] from dup
    """
    dup = int(sys.argv[2])
    bd = sys.argv[1]

    out_file = open('RESULT_COLLECTOR_IGNORE_IEDB_{}.txt'.format(bd), 'w')

    for i in range(dup):
        result_file = os.path.join(BASE_DIR, '{}/dup_{}/weekly_result_METRICS_IGNORE_IEDB_ID_AND_LENGTH.txt'.format(bd, i))
        with open(result_file, 'r') as f:
            for line in f:
                pass
            out_file.write(line)

if __name__ == "__main__":
    collect_result()
    collect_result2()
