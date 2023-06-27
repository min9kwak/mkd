echo "Experiments Started"
SERVER=workstation3
GPUS=3

EPOCHS=30
LEARNING_RATE=0.0001
COSINE_WARMUP=0

BALANCE=True
SAMPLER_TYPE=stratified

DIFFERENT_LR=True
TEMPERATURE=5.0

ALPHA_CE=1.0
ALPHA_KD_REPR=1000.0

STUDENT_PRE="checkpoints/GeneralDistillation-FBP/"
STUDENT_POSITION=last

USE_SPECIFIC=False

# "2023-06-03_04-19-11" "2023-06-10_03-29-36" "2023-06-10_03-29-57" "2023-06-10_13-28-17" "2023-06-10_14-14-27"
for HASH in "2023-06-03_04-19-11"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 30
    do
      for ALPHA_KD_REPR in 1000.0
      do
        for TEMPERATURE in 5.0
        do
          STUDENT_DIR="${STUDENT_PRE}${HASH}"
          python ./run_final_multi.py \
          --gpus $GPUS \
          --server $SERVER \
          --epochs $EPOCHS \
          --batch_size 16 \
          --optimizer adamw \
          --learning_rate $LEARNING_RATE \
          --weight_decay 0.0001 \
          --cosine_warmup $COSINE_WARMUP \
          --balance $BALANCE \
          --sampler_type $SAMPLER_TYPE \
          --alpha_ce $ALPHA_CE \
          --alpha_kd_repr $ALPHA_KD_REPR \
          --different_lr $DIFFERENT_LR \
          --temperature $TEMPERATURE \
          --student_dir $STUDENT_DIR \
          --student_position $STUDENT_POSITION \
          --use_specific $USE_SPECIFIC
        done
      done
    done
  done
done
echo "Finished."
