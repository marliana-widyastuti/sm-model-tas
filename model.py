import utils
import argparse

import time

parser = argparse.ArgumentParser(
    prog='model.py',
    description='Model Training'
)

parser.add_argument('--fileAU', help='filename of CSV containing Australian dataset', default='DF_var_ssm_AUS2.csv')
parser.add_argument('--fileTAS', help='filename of CSV containing Australian dataset', default='DF_var_ssm_TAS.csv')
args, unknown = parser.parse_known_args()

start_time = time.time()

# Au = utils.OneAU(DFAU=args.fileAU)
# Au.run_OneAU()

tas = utils.lstmTAS(DFTAS=args.fileTAS)
# tas.run_lstmTAS()
tas.run_lstmTL()

end_time = time.time()
utils.calculate_execution_time(start_time, end_time)