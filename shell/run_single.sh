echo "Experiments Started"
SERVER=main
GPUS=0

DATA_TYPE=pet


for RANDOM_STATE in {2021..2030}; do
  python ./run_single.py \
  --gpus $GPUS \
  --server $SERVER \
  --data_type $DATA_TYPE \
  --data_file labels/data_info_multi.csv \
  --optimizer adamw
done
echo "Finished."
