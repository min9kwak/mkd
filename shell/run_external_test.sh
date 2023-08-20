echo "Experiments Started"
SERVER=main
GPUS=0

PRETRAINED_TASK=final
EXTERNAL_DATA_TYPE="mri+pib"
HASHS=("2023-07-15_20-25-44")

TRAIN_MODE=test
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.0001

if [ "$PRETRAINED_TASK" == "final" ]; then
    PRETRAINED_FILE_PRE="checkpoints/BestMulti-FBP/"
elif [ "$PRETRAINED_TASK" == "student" ]; then
    PRETRAINED_FILE_PRE="checkpoints/GeneralDistillation-FBP/"
elif [ "$PRETRAINED_TASK" == "multi_demo" ]; then
    PRETRAINED_FILE_PRE="checkpoints/multi_demo/"
elif [ "$PRETRAINED_TASK" == "single_demo" ]; then
    PRETRAINED_FILE_PRE="checkpoints/single_demo/"
else
    echo "Invalid task type: $TASK_TYPE"
    exit 1
fi

# "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08" "2023-06-21_04-19-50"
for HASH in "${HASHS[@]}"; do
  for N_SPLITS in 5; do
    for N_CV in 0; do
      PRETRAINED_DIR="${PRETRAINED_FILE_PRE}${HASH}"

      python ./run_external_test.py \
      --gpus $GPUS \
      --server $SERVER \
      --pretrained_dir $PRETRAINED_DIR \
      --pretrained_task $PRETRAINED_TASK \
      --external_data_type=$EXTERNAL_DATA_TYPE \
      --train_mode $TRAIN_MODE \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --learning_rate $LEARNING_RATE \
      --n_splits $N_SPLITS \
      --n_cv $N_CV
    done
  done
done
echo "Finished."