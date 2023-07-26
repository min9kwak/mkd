echo "Experiments Started"
SERVER=main
GPUS=0

OVERLAP_DIM=10
MISSING_RATE=0.2

TRAIN_LEVEL=-1
LEARNING_RATE_MULTI=0.0001
EPOCHS_MULTI=30
ALPHA_CE=10.0
SHORT=True


#TOTAL_EXP=$((5 * 3 * 1 * 1 * 1 * 1))
#CURRENT_EXP=0

for RANDOM_STATE in {2021..2050}
do
  for MISSING_RATE in 0.1 0.2 0.3 0.4 0.5 0.6
  do
    for OVERLAP_DIM in 10
    do
#      CURRENT_EXP=$((CURRENT_EXP + 1))
#      echo "$CURRENT_EXP / $TOTAL_EXP"
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
      --short $SHORT
    done
  done
done
echo "Finished."
