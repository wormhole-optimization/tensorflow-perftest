#!/bin/bash

M=100
N=10
R=5
SPARSITY=0.01
python3 gendata_pnmf.py 'data/pnmf_X_'$M'_'$N'_d'$SPARSITY'.npz' 'data/pnmf_W_'$M'_'$R'.npz' 'data/pnmf_H_'$R'_'$N'.npz' $M $N $R $SPARSITY
