echo "Experiments Started"
SERVER=main
GPUS=0

OVERLAP_DIM=10
MISSING_RATE=0.2

TRAIN_LEVEL=3

#TOTAL_EXP=$((5 * 3 * 1 * 1 * 1 * 1))
#CURRENT_EXP=0

for RANDOM_STATE in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
do
  for MISSING_RATE in 0.2 0.4 0.6 0.8
  do
    for OVERLAP_DIM in 20 15 10 5 0
    do
#      CURRENT_EXP=$((CURRENT_EXP + 1))
#      echo "$CURRENT_EXP / $TOTAL_EXP"
      python ./run_simulator.py \
      --server $SERVER \
      --gpus $GPUS \
      --overlap_dim $OVERLAP_DIM \
      --missing_rate $MISSING_RATE \
      --random_state $RANDOM_STATE \
      --train_level $TRAIN_LEVEL

    done
  done
done
echo "Finished."
