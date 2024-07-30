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
N_INCOMPLETE=1000
N_VALIDATION=1000
N_TEST=1000

HIDDEN=10
SIMPLE=False

ALPHA_CE=1
ALPHA_SIM_SMT=2
ALPHA_SIM_FINAL=2
ALPHA_DIFF=1

ALPHA_RECON=10
ALPHA_KD_CLF=10
ALPHA_KD_REPR=10

TRAIN_LEVEL="1,2"

for RANDOM_STATE in {2021..2050}; do
  for ALPHA_RECON in 1 10 50 100 200; do
    for ALPHA_KD_CLF in 1 10 50 100 200; do
      for ALPHA_KD_REPR in 1 2 4 5 10; do
        ALPHA_SIM_FINAL=$ALPHA_SIM_SMT
        NOTE="hyperparameter"

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
        --n_validation $N_VALIDATION \
        --n_test $N_TEST \
        --simple $SIMPLE \
        --hidden $HIDDEN \
        --alpha_ce $ALPHA_CE \
        --alpha_sim_smt $ALPHA_SIM_SMT \
        --alpha_sim_final $ALPHA_SIM_FINAL \
        --alpha_diff $ALPHA_DIFF \
        --alpha_recon $ALPHA_RECON \
        --alpha_kd_clf $ALPHA_KD_CLF \
        --alpha_kd_repr $ALPHA_KD_REPR \
        --random_state $RANDOM_STATE \
        --note "$NOTE" \
        --train_level $TRAIN_LEVEL
      done
    done
  done
done
echo "Finished."