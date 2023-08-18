echo "Experiments Started"
SERVER=workstation3
GPUS=5

TASK_TYPE=single
HIDDEN_DEMO="4,4"
DIFFERENT_LR_DEMO=False

EPOCHS=10
LEARNING_RATE=0.0001

if [ "$TASK_TYPE" == "single" ]; then
    PRETRAINED_FILE_PRE="checkpoints/BestMulti-FBP"

    # specific=True
    HASHS=("2023-07-28_20-59-35" "2023-07-28_20-09-46" "2023-07-28_07-48-56" "2023-07-28_06-53-09" "2023-07-27_22-33-06" "2023-07-27_21-38-00" "2023-07-27_13-02-05" "2023-07-27_12-16-37" "2023-07-26_22-49-18" "2023-07-26_22-48-40")

    # specific=False
    # HASHS=("2023-07-18_18-11-27" "2023-07-18_18-11-02" "2023-07-18_18-10-27" "2023-07-16_12-29-07" "2023-07-15_20-25-44" "2023-07-14_19-14-39" "2023-07-14_03-23-07" "2023-07-13_11-21-04" "2023-07-12_16-17-02" "2023-07-11_17-07-33")
elif [ "$TASK_TYPE" == "multi" ]; then
    PRETRAINED_FILE_PRE="checkpoints/GeneralDistillation-FBP"
    HASHS=("2023-07-11_10-05-11" "2023-07-11_01-43-57" "2023-07-10_17-01-33" "2023-07-10_07-46-21" "2023-07-09_22-29-10" "2023-06-21_04-19-50" "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08")
else
    echo "Invalid task type: $TASK_TYPE"
    exit 1
fi

# "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08" "2023-06-21_04-19-50"
for HASH in "${HASHS[@]}"; do
  for LEARNING_RATE in 0.001 0.0001; do
    for DIFFERENT_LR_DEMO in True False; do
      for HIDDEN_DEMO in "4,4" "3"; do
        for EPOCHS in 10 30; do
          PRETRAINED_DIR="${PRETRAINED_FILE_PRE}${HASH}"
          python ./run_final_multi.py \
          --gpus $GPUS \
          --server $SERVER \
          --pretrained_dir $PRETRAINED_DIR \
          --task_type $TASK_TYPE \
          --epochs $EPOCHS \
          --optimizer adamw \
          --learning_rate $LEARNING_RATE \
          --weight_decay 0.0001 \
          --cosine_warmup 0 \
          --different_lr_demo $DIFFERENT_LR_DEMO \
          --hidden_demo=$HIDDEN_DEMO
        done
      done
    done
  done
done
echo "Finished."