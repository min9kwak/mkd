echo "Experiments Started"
SERVER=main
GPUS=0

EPOCHS=10
LEARNING_RATE=0.0001
COSINE_WARMUP=-1

BALANCE=True
SAMPLER_TYPE=stratified

TEMPERATURE=1.0
ALPHA_RECON=0.5
ALPHA_KD_CLF=1.0

TEACHER_PRE="checkpoints/GeneralTeacher-FBP/"
HASH="2023-05-19_20-47-05"
TEACHER_POSITION=last
USE_TEACHER=True

# "2023-05-16_03-04-13" "2023-05-14_22-07-48" "2023-05-14_22-07-13" "2023-05-12_14-28-19"
for HASH in "2023-05-16_03-04-13" "2023-05-14_22-07-48" "2023-05-14_22-07-13"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 10 30
    do
      for ALPHA_KD_CLF in 0.5 1.0
      do
        for TEMPERATURE in 1.0 2.0
        do
          TEACHER_DIR="${TEACHER_PRE}${HASH}"
          python ./run_general_teacher.py \
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
          --alpha_recon $ALPHA_RECON \
          --alpha_kd_clf $ALPHA_KD_CLF \
          --temperature $TEMPERATURE \
          --teacher_dir $TEACHER_DIR \
          --teacher_position $TEACHER_POSITION \
          --use_teacher $USE_TEACHER
        done
      done
    done
  done
done
echo "Finished."
