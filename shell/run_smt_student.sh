echo "Experiments Started"
SERVER=workstation3
GPUS=3

EPOCHS=30
LEARNING_RATE=0.0001
COSINE_WARMUP=0

BALANCE=True
SAMPLER_TYPE=stratified

DIFFERENT_LR=False
TEMPERATURE=5.0

ALPHA_CE=1.0
ALPHA_KD_CLF=100.0
ALPHA_KD_REPR=0.0

TEACHER_PRE="checkpoints/SMT-FBP/"  # Update to your SMT checkpoint directory
TEACHER_POSITION=last

USE_TEACHER=True
USE_SPECIFIC=False
INHERIT_CLASSIFIER=True

# Example hashes - replace with your actual SMT checkpoint timestamps
for HASH in "2023-06-03_04-19-11"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 30
    do
      for ALPHA_KD_CLF in 100.0
      do
        for TEMPERATURE in 5.0
        do
          TEACHER_DIR="${TEACHER_PRE}${HASH}"
          python ./run_smt_student.py \
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
          --alpha_kd_clf $ALPHA_KD_CLF \
          --alpha_kd_repr $ALPHA_KD_REPR \
          --different_lr $DIFFERENT_LR \
          --temperature $TEMPERATURE \
          --teacher_dir $TEACHER_DIR \
          --teacher_position $TEACHER_POSITION \
          --use_teacher $USE_TEACHER \
          --use_specific $USE_SPECIFIC \
          --inherit_classifier $INHERIT_CLASSIFIER
        done
      done
    done
  done
done
echo "Finished."
