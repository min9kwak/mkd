echo "Experiments Started"
SERVER=main
GPUS=0

N_TRAIN=1000
N_TEST=1000
X_DIM=50
XS_DIM=20
OVERLAP_DIM=10
HYPERPLANE_DIM=500
MISSING_RATE=0.2

OPTIMIZER=adamw
BATCH_SIZE=16

EPOCHS_TEACHER=100
LEARNING_RATE_TEACHER=0.001
WEIGHT_DECAY_TEACHER=0.0001

EPOCHS_KD=30
LEARNING_RATE_KD=0.00001
WEIGHT_DECAY_KD=0.0001

EPOCHS_FINAL=100
LEARNING_RATE_FINAL=0.001
WEIGHT_DECAY_FINAL=0.0001

COSINE_CYCLES=1
COSINE_WARMUP=0

HIDDEN=25

SAVE_LOG=True

ALPHA_CE=10.0
ALPHA_SIM=7.0
ALPHA_DIFF=5.0
ALPHA_RECON=0.1
ALPHA_KD_CLF=100.0
ALPHA_KD_REPR=500.0

TEMPERATURE=5.0

RANDOM_STATE=2021

TRAIN_LEVEL=1

TOTAL_EXP=$((5 * 3 * 1 * 1 * 1 * 1))
CURRENT_EXP=0

for RANDOM_STATE in 2021 2022 2023 2024 2025
do
  for ALPHA_SIM in 1.0 3.0 5.0
  do
    for ALPHA_CE in 1.0 3.0 5.0 10.0
    do
      for LEARNING_RATE_KD in 0.0001
      do
        for ALPHA_RECON in 100.0 200.0 500.0
        do
          for COSINE_WARMUP in 0
          do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            echo "$CURRENT_EXP / $TOTAL_EXP"

            python ./run_simulator.py \
            --server $SERVER \
            --gpus $GPUS \
            --n_train $N_TRAIN \
            --n_test $N_TEST \
            --x_dim $X_DIM \
            --xs_dim $XS_DIM \
            --overlap_dim $OVERLAP_DIM \
            --hyperplane_dim $HYPERPLANE_DIM \
            --missing_rate $MISSING_RATE \
            --optimizer $OPTIMIZER \
            --batch_size $BATCH_SIZE \
            --epochs_teacher $EPOCHS_TEACHER \
            --learning_rate_teacher $LEARNING_RATE_TEACHER \
            --weight_decay_teacher $WEIGHT_DECAY_TEACHER \
            --cosine_cycles $COSINE_CYCLES \
            --cosine_warmup $COSINE_WARMUP \
            --epochs_kd $EPOCHS_KD \
            --learning_rate_kd $LEARNING_RATE_KD \
            --weight_decay_kd $WEIGHT_DECAY_KD \
            --epochs_final $EPOCHS_FINAL \
            --learning_rate_final $LEARNING_RATE_FINAL \
            --weight_decay_final $WEIGHT_DECAY_FINAL \
            --hidden $HIDDEN \
            --save_log $SAVE_LOG \
            --alpha_ce $ALPHA_CE \
            --alpha_sim $ALPHA_SIM \
            --alpha_diff $ALPHA_DIFF \
            --alpha_recon $ALPHA_RECON \
            --alpha_kd_clf $ALPHA_KD_CLF \
            --alpha_kd_repr $ALPHA_KD_REPR \
            --temperature $TEMPERATURE \
            --random_state $RANDOM_STATE \
            --train_level $TRAIN_LEVEL

          done
        done
      done
    done
  done
done
echo "Finished."
