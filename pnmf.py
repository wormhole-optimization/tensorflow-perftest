import sys
import numpy as np
import tensorflow as tf
from scipy.sparse import save_npz
from scipy.sparse import load_npz

coo = load_npz(sys.argv[1])
indices = np.mat([coo.row, coo.col]).transpose()
X = tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape)))
coo = load_npz(sys.argv[2])
indices = np.mat([coo.row, coo.col]).transpose()
W = tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape)))
coo = load_npz(sys.argv[3])
indices = np.mat([coo.row, coo.col]).transpose()
H = tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape)))

# k = sys.argv[4]
eps = float(sys.argv[4])
max_iter = int(sys.argv[5])
iter = 1;

while( iter < max_iter ):
   # H = (H*(t(W)%*%(X/(W%*%H+eps)))) / t(colSums(W));
   H_1 = tf.math.multiply(                       \
             H,                                  \
             tf.linalg.matmul(                   \
                 tf.linalg.matrix_transpose(W),  \
                 tf.math.divide(                 \
                     X,                          \
                     tf.math.add(                \
                         tf.linalg.matmul(W, H), \
                         eps))))
   H_2 = tf.linalg.matrix_transpose(tf.math.reduce_sum(W, axis=1, keepdims=True))
   H = tf.math.divide(H_1, H_2)

   # W = (W*((X/(W%*%H+eps))%*%t(H))) / t(rowSums(H));
   W = tf.math.divide(                              \
           tf.math.multiply(                        \
               W,                                   \
               tf.linalg.matmul(                    \
                   tf.math.divide(                  \
                       X,                           \
                       tf.math.add(                 \
                           tf.linalg.matmul(W, H),  \
                           eps)),                   \
                   tf.linalg.matrix_transpose(H))), \
           tf.linalg.matrix_transpose(              \
               tf.math.reduce_sum(H, axis=0, keepdims=True)))

   # obj = sum(W%*%H) - sum(X*log(W%*%H+eps));
   obj = tf.math.subtract(                           \
             tf.math.reduce_sum(                     \
                 tf.linalg.matmul(W, H)),            \
             tf.math.reduce_sum(                     \
                 tf.math.multiply(                   \
                     X,                              \
                     tf.math.log(                    \
                         tf.math.add(                \
                             tf.linalg.matmul(W, H), \
                             eps)))))
   
   print("iter=" + str(iter) + " obj=" + str(obj))
   
   iter = iter + 1

np.save(sys.argv[6], tf.make_ndarray(W))
np.save(sys.argv[7], tf.make_ndarray(H))

