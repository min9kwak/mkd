echo "Experiments Started"
SERVER=main
GPUS=0

MRI_TYPE=template
PET_TYPE=FBP
IMAGE_SIZE=72
CROP_SIZE=64
RANDOM_STATE=2023

INTENSITY=scale

ENCODER_TYPE=resnet50

EPOCHS=100
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.001

TRAIN_SLICES=random
NUM_SLICES=5
SLICE_RANGE=0.15

WARMUP=0
FEATURE_KD=cos
TEMPERATURE=1.0
ALPHA_T2S=3.0
ALPHA_S2T=3.0


for WARMUP in 0 20
do
	for RANDOM_STATE in 2021 2022 2023
	do
	  for FEATURE_KD in mse
	  do
      python ./run_distillation.py \
      --gpus $GPUS \
      --server $SERVER \
      --pet_type $PET_TYPE \
      --data_file labels/data_info_multi.csv \
      --mri_type $MRI_TYPE \
      --image_size_mri $IMAGE_SIZE \
      --intensity_mri $INTENSITY \
      --crop_size_mri $CROP_SIZE \
      --rotate_mri \
      --flip_mri \
      --image_size_pet $IMAGE_SIZE \
      --intensity_pet $INTENSITY \
      --crop_size_pet $CROP_SIZE \
      --rotate_pet \
      --flip_pet \
      --prob 0.5 \
      --train_slices $TRAIN_SLICES \
      --num_slices $NUM_SLICES \
      --slice_range $SLICE_RANGE \
      --encoder_type $ENCODER_TYPE \
      --small_kernel \
      --random_state $RANDOM_STATE \
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
      --mixed_precision \
      --add_type add \
      --warmup $WARMUP \
      --feature_kd $FEATURE_KD \
      --temperature $TEMPERATURE \
      --alpha_t2s $ALPHA_T2S \
      --alpha_s2t $ALPHA_S2T
    done
	done
done
echo "Finished."
