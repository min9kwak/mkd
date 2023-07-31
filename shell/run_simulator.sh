echo "Experiments Started"
SERVER=main
GPUS=0

OVERLAP_DIM=10
MISSING_RATE=0.2

TRAIN_LEVEL=1
LEARNING_RATE_MULTI=0.0001
EPOCHS_MULTI=30
ALPHA_CE=10.0

SHORT=False
SIMPLE=True

ALPHA_SIM=10.0
ALPHA_KD_CLF=50.0

#TOTAL_EXP=$((5 * 3 * 1 * 1 * 1 * 1))
#CURRENT_EXP=0

for RANDOM_STATE in {2021..2030}
do
  for ALPHA_DIFF in 10
  do
    for ALPHA_SIM in 10 12 15 17 20
    do
      for ALPHA_RECON in 1 10 20 50 100 200
      do
        for ALPHA_CE in 1 2 5 10 12 15 20
        do
#         CURRENT_EXP=$((CURRENT_EXP + 1))
#         echo "$CURRENT_EXP / $TOTAL_EXP"
          python ./run_simulator.py \
          --server $SERVER \
          --gpus $GPUS \
          --overlap_dim $OVERLAP_DIM \
          --missing_rate $MISSING_RATE \
          --random_state $RANDOM_STATE \
          --alpha_ce $ALPHA_CE \
          --train_level $TRAIN_LEVEL \
          --learning_rate_multi $LEARNING_RATE_MULTI \
          --epochs_multi $EPOCHS_MULTI \
          --alpha_sim $ALPHA_SIM \
          --alpha_recon $ALPHA_RECON \
          --alpha_diff $ALPHA_DIFF \
          --alpha_kd_clf $ALPHA_KD_CLF \
          --short $SHORT \
          --simple $SIMPLE
        done
      done
    done
  done
done
echo "Finished."
