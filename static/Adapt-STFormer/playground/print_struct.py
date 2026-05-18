import sys

sys.path.insert(0, "../")
from datasets import parse_db_struct, print_db_concise

db = parse_db_struct("structFiles/nordland_train_d-40_d2-10.db")
print_db_concise(db)
