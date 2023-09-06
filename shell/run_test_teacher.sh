echo "Experiments Started"
SERVER=main
GPUS=0

EPOCHS_SMT_STUDENT=30
EPOCHS_MULTI_STUDENT=30

for RANDOM_STATE in {2021..2030}; do
  for MISSING_RATE in 0.1 0.2 0.3 0.4 0.5 0.6; do
  python ./run_test_teacher.py \
  --server $SERVER \
  --gpus $GPUS \
  --missing_rate $MISSING_RATE \
  --epochs_smt_student $EPOCHS_SMT_STUDENT \
  --epochs_multi_student $EPOCHS_MULTI_STUDENT \
  --random_state $RANDOM_STATE
  done
done
echo "Finished."
