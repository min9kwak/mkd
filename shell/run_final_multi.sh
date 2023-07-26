echo "Experiments Started"
SERVER=workstation3
GPUS=5

EPOCHS=100
LEARNING_RATE=0.001
COSINE_WARMUP=0

BALANCE=True
SAMPLER_TYPE=stratified

DIFFERENT_LR=False
TEMPERATURE=5.0

ALPHA_CE=1.0
ALPHA_KD_REPR=500.0

STUDENT_PRE="checkpoints/GeneralDistillation-FBP/"
STUDENT_POSITION=last

USE_SPECIFIC_FINAL=True
USE_TEACHER=False
USE_STUDENT=True

# "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08" "2023-06-21_04-19-50"
for HASH in "2023-06-18_09-50-44" "2023-06-18_06-20-15" "2023-06-18_02-19-54" "2023-06-17_22-36-08" "2023-06-21_04-19-50"
do
  for LEARNING_RATE in 0.001
  do
    for EPOCHS in 100
    do
      for ALPHA_KD_REPR in 500.0
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
          --use_specific_final $USE_SPECIFIC_FINAL \
          --use_teacher $USE_TEACHER \
          --use_student $USE_STUDENT
        done
      done
    done
  done
done
echo "Finished."
