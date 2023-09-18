echo "Experiments Started"
SERVER=main
GPUS=0

ZS_DIM=10
Z1_DIM=10
Z2_DIM=10

RHO=0.5
SIGMA=1.0

XS_DIM=20
X1_DIM=20
X2_DIM=20

MU_0=0.0
MU_1=0.5

N_COMPLETE=1000
N_INCOMPLETE=0
N_TEST=1000

HIDDEN=10

ALPHA_CE=1.0
ALPHA_SIM=10.0
ALPHA_DIFF=5.0
ALPHA_RECON=100.0
ALPHA_KD_CLF=100.0
ALPHA_KD_REPR=100.0

for RANDOM_STATE in {2021..2040}; do
  for ALPHA_CE in 1.0 2.0 5.0 7.0 10.0; do
    for ALPHA_SIM in 1.0 2.0 5.0 7.0 10.0; do
      for ALPHA_DIFF in 1.0 2.0 5.0 7.0 10.0; do
        python ./run_simulator.py \
        --server $SERVER \
        --gpus $GPUS \
        --zs_dim $ZS_DIM \
        --z1_dim $Z1_DIM \
        --z2_dim $Z2_DIM \
        --rho $RHO \
        --sigma $SIGMA \
        --xs_dim $XS_DIM \
        --x1_dim $X1_DIM \
        --x2_dim $X2_DIM \
        --mu_0 $MU_0 \
        --mu_1 $MU_1 \
        --n_complete $N_COMPLETE \
        --n_incomplete $N_INCOMPLETE \
        --n_test $N_TEST \
        --hidden $HIDDEN \
        --alpha_ce $ALPHA_CE \
        --alpha_sim $ALPHA_SIM \
        --alpha_diff $ALPHA_DIFF \
        --alpha_recon $ALPHA_RECON \
        --alpha_kd_clf $ALPHA_KD_CLF \
        --alpha_kd_repr $ALPHA_KD_REPR \
        --random_state $RANDOM_STATE
      done
    done
  done
done
echo "Finished."
