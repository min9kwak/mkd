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

N_COMPLETE=500
N_INCOMPLETE=500
N_TEST=1000

HIDDEN=10
SIMPLE=False

ALPHA_CE=1.0
ALPHA_DISC=1.0
ALPHA_RECON=100.0
ALPHA_KD_CLF=100.0
ALPHA_KD_REPR=100.0


for RANDOM_STATE in {2021..2050}; do
  for ALPHA_DISC in 1.0; do
    for N_INCOMPLETE in 200 300 500; do
      for ZS_DIM in 2 6 10 14 18 22 26; do
        NOTE="maintain total dim"
        N_COMPLETE=$N_INCOMPLETE
        Z1_DIM=$(((30 - ZS_DIM) / 2))
        Z2_DIM=$Z1_DIM
        ALPHA_KD_CLF=$ALPHA_RECON
        ALPHA_KD_REPR=$ALPHA_KD_CLF
        python ./run_simulator_disc.py \
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
        --simple $SIMPLE \
        --hidden $HIDDEN \
        --alpha_ce $ALPHA_CE \
        --alpha_disc $ALPHA_DISC \
        --alpha_recon $ALPHA_RECON \
        --alpha_kd_clf $ALPHA_KD_CLF \
        --alpha_kd_repr $ALPHA_KD_REPR \
        --random_state $RANDOM_STATE \
        --note "$NOTE"
      done
    done
  done
done
echo "Finished."


for RANDOM_STATE in {2021..2050}; do
  for ALPHA_DISC in 1.0; do
    for N_INCOMPLETE in 200 300 500; do
      for ZS_DIM in 2 6 10 14 18; do
        NOTE="maintain modality dim"
        N_COMPLETE=$N_INCOMPLETE
        Z1_DIM=$((20 - ZS_DIM))
        Z2_DIM=$Z1_DIM
        ALPHA_KD_CLF=$ALPHA_RECON
        ALPHA_KD_REPR=$ALPHA_KD_CLF
        python ./run_simulator_disc.py \
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
        --simple $SIMPLE \
        --hidden $HIDDEN \
        --alpha_ce $ALPHA_CE \
        --alpha_disc $ALPHA_DISC \
        --alpha_recon $ALPHA_RECON \
        --alpha_kd_clf $ALPHA_KD_CLF \
        --alpha_kd_repr $ALPHA_KD_REPR \
        --random_state $RANDOM_STATE \
        --note "$NOTE"
      done
    done
  done
done
echo "Finished."
