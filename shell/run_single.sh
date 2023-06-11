echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
MCI_ONLY=True
USE_UNLABELED=False
TRAIN_SLICES=fixed
SPACE=3
N_POINTS=5
MISSING_RATE=-1

EXTRACTOR_TYPE=resnet50
SMALL_KERNEL=True

EPOCHS=100
LEARNING_RATE=0.001
COSINE_WARMUP=0

RANDOM_STATE=2021

BALANCE=True
SAMPLER_TYPE=stratified

for RANDOM_STATE in 2021
do
	for EPOCHS in 100
	do
	  for DIFFERENT_LR in True False
	  do
      python ./run_single.py \
      --gpus $GPUS \
      --server $SERVER \
      --pet_type $PET_TYPE \
      --mci_only $MCI_ONLY \
      --use_unlabeled $USE_UNLABELED \
      --data_file labels/data_info_multi.csv \
      --missing_rate $MISSING_RATE \
      --crop_size_mri $CROP_SIZE \
      --train_slices $TRAIN_SLICES \
      --space $SPACE \
      --n_points $N_POINTS \
      --extractor_type $EXTRACTOR_TYPE \
      --small_kernel $SMALL_KERNEL \
      --random_state $RANDOM_STATE \
      --epochs $EPOCHS \
      --batch_size 16 \
      --optimizer adamw \
      --learning_rate $LEARNING_RATE \
      --weight_decay 0.0001 \
      --cosine_warmup $COSINE_WARMUP \
      --balance $BALANCE \
      --sampler_type $SAMPLER_TYPE
    done
	done
done
echo "Finished."
