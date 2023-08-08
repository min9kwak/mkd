echo "Experiments Started"
SERVER=main
GPUS=0

OVERLAP_DIM=10
MISSING_RATE=0.2

TRAIN_LEVEL=2
LEARNING_RATE_MULTI=0.0001
EPOCHS_MULTI=30
ALPHA_CE=10.0

SHORT=False
SIMPLE=True
ENCODER_ACT=None

ALPHA_CE=1.0
ALPHA_SIM=2.0
ALPHA_KD_CLF=50.0

#TOTAL_EXP=$((5 * 3 * 1 * 1 * 1 * 1))
#CURRENT_EXP=0

for RANDOM_STATE in {2021..2030}
do
  for ALPHA_RECON in 0.1 0.5 1.0 2.0 5.0
  do
    for ALPHA_DIFF in 1.0 2.0 5.0 10.0
    do
      for ALPHA_SIM in 1.0 2.0 5.0 7.0 10.0
      do
        for ALPHA_CE in 1.0 2.0 5.0 10.0
        do
          for LEARNING_RATE_TEACHER in 0.001 0.0001 0.0005
          do
            python ./run_simulator.py \
            --server $SERVER \
            --gpus $GPUS \
            --overlap_dim $OVERLAP_DIM \
            --missing_rate $MISSING_RATE \
            --train_level $TRAIN_LEVEL \
            --learning_rate_teacher $LEARNING_RATE_TEACHER \
            --learning_rate_multi $LEARNING_RATE_MULTI \
            --epochs_multi $EPOCHS_MULTI \
            --alpha_ce $ALPHA_CE \
            --alpha_sim $ALPHA_SIM \
            --alpha_diff $ALPHA_DIFF \
            --alpha_recon $ALPHA_RECON \
            --alpha_kd_clf $ALPHA_KD_CLF \
            --short $SHORT \
            --simple $SIMPLE \
            --encoder_act $ENCODER_ACT \
            --random_state $RANDOM_STATE
          done
        done
      done
    done
  done
done
echo "Finished."
