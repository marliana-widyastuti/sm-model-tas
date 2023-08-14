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

M = utils.model(DFAU=args.fileAU, DFTAS=args.fileTAS)
print(M.stations)

M.run_OneAU()
M.run_lstmTAS()
M.run_lstmTL()

M.run_mlpOneAU()
M.run_mlpTAS()
M.run_mlpTL()

end_time = time.time()
utils.calculate_execution_time(start_time, end_time)