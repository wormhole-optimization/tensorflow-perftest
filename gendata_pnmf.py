import sys
import numpy as np
from scipy.sparse import random
from scipy.sparse import save_npz

x_file = sys.argv[1]
w_file = sys.argv[2]
h_file = sys.argv[3]
m = int(sys.argv[4])
n = int(sys.argv[5])
r = int(sys.argv[6])
sparsity = float(sys.argv[7])

np.random.seed(42069)

w = random(m, r, density=1)
h = random(r, n, density=1)
x = random(m, n, density=sparsity)

save_npz(x_file, x)
save_npz(w_file, w)
save_npz(h_file, h)
