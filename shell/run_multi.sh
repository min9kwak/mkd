echo "Experiments Started"
SERVER=main
GPUS=0

DATA_TYPE=multi
PET_TYPE=FBP
IMAGE_SIZE=72
CROP_SIZE=64
RANDOM_STATE=2023

INTENSITY=scale

EPOCHS=100
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.0001

TRAIN_SLICES=random
NUM_SLICES=5
SLICE_RANGE=0.15

for RANDOM_STATE in 2021 2022 2023
do
	for LEARNING_RATE in 0.001 0.0001
	do
	  for ENCODER_TYPE in resnet50 densenet121
	  do
      python ./run_multi.py \
      --gpus $GPUS \
      --server $SERVER \
      --data_type $DATA_TYPE \
      --pet_type $PET_TYPE \
      --root D:/data/ADNI \
      --data_file labels/data_info_multi.csv \
      --image_size $IMAGE_SIZE \
      --crop_size $CROP_SIZE \
      --rotate \
      --flip \
      --affine \
      --prob 0.5 \
      --train_slices $TRAIN_SLICES \
      --num_slices $NUM_SLICES \
      --slice_range $SLICE_RANGE \
      --encoder_type $ENCODER_TYPE \
      --small_kernel \
      --add_type add \
      --random_state $RANDOM_STATE \
      --intensity $INTENSITY \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --optimizer $OPTIMIZER \
      --learning_rate $LEARNING_RATE \
      --weight_decay 0.0001 \
      --cosine_warmup 0 \
      --cosine_cycles 1 \
      --cosine_min_lr 0.0 \
      --save_every 1000 \
      --enable_wandb \
      --balance
    done
	done
done
echo "Finished."
