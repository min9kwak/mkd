echo "Experiments Started"
SERVER=main
GPUS=0

DATA_TYPE=mri
MRI_TYPE=individual
PET_TYPE=FBP
IMAGE_SIZE=72
CROP_SIZE=64
RANDOM_STATE=2023

INTENSITY=simple

EPOCHS=100
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.001

TRAIN_SLICES=random
NUM_SLICES=5
SLICE_RANGE=0.15

for RANDOM_STATE in 2021 2022 2023
do
	for DATA_TYPE in mri
	do
	  for ENCODER_TYPE in resnet50
	  do
      python ./run_single.py \
      --gpus $GPUS \
      --server $SERVER \
      --data_type $DATA_TYPE \
      --mri_type $MRI_TYPE \
      --pet_type $PET_TYPE \
      --data_file labels/data_info_multi.csv \
      --pet_type FBP \
      --image_size $IMAGE_SIZE \
      --crop_size $CROP_SIZE \
      --rotate \
      --flip \
      --prob 0.5 \
      --train_slices $TRAIN_SLICES \
      --num_slices $NUM_SLICES \
      --slice_range $SLICE_RANGE \
      --encoder_type $ENCODER_TYPE \
      --small_kernel \
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
      --balance \
	  --mixed_precision
    done
	done
done
echo "Finished."
