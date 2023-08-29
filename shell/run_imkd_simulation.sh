echo "Experiments Started"
SERVER=main
GPUS=0

OVERLAP_DIM=10
MISSING_RATE=0.2

TRAIN_LEVEL="1,2,3,4"

ENCODER_ACT=relu

EPOCHS_IMKD=50
LEARNING_RATE_IMKD=0.00001
LEARNING_RATE_IMKD_S=0.0001

for RANDOM_STATE in {2021..2050}; do
  for MISSING_RATE in 0.1 0.2 0.3 0.4 0.5 0.6; do

    python ./run_imkd_simulation.py \
    --server $SERVER \
    --gpus $GPUS \
    --overlap_dim $OVERLAP_DIM \
    --missing_rate $MISSING_RATE \
    --train_level=$TRAIN_LEVEL \
    --epochs_imkd $EPOCHS_IMKD \
    --learning_rate_imkd $LEARNING_RATE_IMKD \
    --learning_rate_imkd_s $LEARNING_RATE_IMKD_S \
    --encoder_act $ENCODER_ACT \
    --random_state $RANDOM_STATE

  done
done
echo "Finished."
