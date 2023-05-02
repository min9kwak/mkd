echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
TRAIN_SLICES=fixed
MISSING_RATE=-1

ENCODER_TYPE=resnet50
SMALL_KERNEL=True

HIDDEN=128
SWAP=False
MLP=True
DROPOUT=0.0

EPOCHS=100
LEARNING_RATE=0.001
COSINE_WARMUP=0

CE_ONLY=False

ALPHA_SIM=1.0
ALPHA_RECON=1.0

RANDOM_STATE=2021

USE_SIM=True

for RANDOM_STATE in 2021
do
  for CE_ONLY in False
  do
    for SWAP in False True
    do
      for USE_SPECIFIC in False
      do
        for ADD_TYPE in add
        do
          python ./run_swap.py \
          --gpus $GPUS \
          --server $SERVER \
          --pet_type $PET_TYPE \
          --data_file labels/data_info_multi.csv \
          --missing_rate $MISSING_RATE \
          --crop_size_mri $CROP_SIZE \
          --crop_size_pet $CROP_SIZE \
          --train_slices $TRAIN_SLICES \
          --extractor_type $ENCODER_TYPE \
          --small_kernel $SMALL_KERNEL \
          --random_state $RANDOM_STATE \
          --hidden $HIDDEN \
          --swap $SWAP \
          --mlp $MLP \
          --dropout $DROPOUT \
          --epochs $EPOCHS \
          --batch_size 16 \
          --optimizer adamw \
          --learning_rate $LEARNING_RATE \
          --weight_decay 0.0001 \
          --use_specific $USE_SPECIFIC \
          --use_sim $USE_SIM \
          --add_type $ADD_TYPE \
          --ce_only $CE_ONLY \
          --cosine_warmup $COSINE_WARMUP \
          --alpha_sim $ALPHA_SIM \
          --alpha_recon $ALPHA_RECON
        done
      done
    done
  done
done
echo "Finished."
