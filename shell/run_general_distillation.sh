echo "Experiments Started"
SERVER=workstation3
GPUS=0

EPOCHS=10
LEARNING_RATE=0.0001
COSINE_WARMUP=0

BALANCE=True
SAMPLER_TYPE=stratified

DIFFERENT_LR=True
TEMPERATURE=1.0

ALPHA_CE=1.0
ALPHA_RECON=0.5
ALPHA_KD_CLF=10.0

TEACHER_PRE="checkpoints/GeneralTeacher-FBP/"
HASH="2023-05-31_02-42-45"
TEACHER_POSITION=last
USE_TEACHER=True

# "2023-06-03_15-08-06" "2023-06-03_04-19-11" "2023-06-03_04-18-56" "2023-06-02_13-39-06"
for HASH in "2023-06-03_15-08-06" "2023-06-03_04-19-11" "2023-06-03_04-18-56" "2023-06-02_13-39-06"
do
  for LEARNING_RATE in 0.001
  do
    for EPOCHS in 100
    do
      for ALPHA_KD_CLF in 10.0
      do
        for TEMPERATURE in 3.0 5.0
        do
          TEACHER_DIR="${TEACHER_PRE}${HASH}"
          python ./run_general_distillation.py \
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
          --alpha_recon $ALPHA_RECON \
          --alpha_kd_clf $ALPHA_KD_CLF \
          --different_lr $DIFFERENT_LR \
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
