echo "Experiments Started"
SERVER=workstation3
GPUS=5

EPOCHS=10
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.00001

RANDOM_STATE=2021
N_SPLITS=5
N_CV=0
TRAIN_MODE="train"

STUDENT_PRE="checkpoints/GeneralDistillation-FBP/"
STUDENT_POSITION=last

for HASH in "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08" "2023-06-21_04-19-50" "2023-07-09_22-29-10" "2023-07-10_07-46-21" "2023-07-10_17-01-33" "2023-07-11_01-43-57" "2023-07-11_10-05-11"
do
  for LEARNING_RATE in 0.00001 0.0001
  do
    for N_CV in 0
    do
      STUDENT_DIR="${STUDENT_PRE}${HASH}"
      python ./run_aibl_cv.py \
      --gpus $GPUS \
      --server $SERVER \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --optimizer $OPTIMIZER \
      --learning_rate $LEARNING_RATE \
      --weight_decay 0 \
      --cosine_warmup 0 \
      --cosine_cycles 1 \
      --cosine_min_lr 0.0 \
      --save_every 2000 \
      --enable_wandb \
      --student_dir $STUDENT_DIR \
      --student_position $STUDENT_POSITION \
      --balance \
      --random_state $RANDOM_STATE \
      --n_splits $N_SPLITS \
      --n_cv $N_CV \
      --train_mode $TRAIN_MODE
    done
  done
done
echo "Finished."
