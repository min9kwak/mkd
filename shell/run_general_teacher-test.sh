echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
TRAIN_SLICES=sagittal
MISSING_RATE=-1

ENCODER_TYPE=resnet50
SMALL_KERNEL=True

HIDDEN=64
SWAP=False
MLP=False
DROPOUT=0.0

EPOCHS=100
LEARNING_RATE=0.001
COSINE_WARMUP=0

CE_ONLY=False
WARMUP=-1
LOSS_DIFF="mse"

ALPHA_SIM=0.1
ALPHA_DIFF=0.1
ALPHA_RECON=0.1

RANDOM_STATE=2021


for RANDOM_STATE in 2021
do
  for USE_PROJECTOR in True
  do
    for USE_SPECIFIC in True
    do
      for USE_TRANSFORMER in False
      do
        for ADD_TYPE in add
        do
          python ./run_general_teacher.py \
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
          --loss_diff $LOSS_DIFF \
          --mlp $MLP \
          --dropout $DROPOUT \
          --epochs $EPOCHS \
          --batch_size 16 \
          --optimizer adamw \
          --learning_rate $LEARNING_RATE \
          --weight_decay 0.0001 \
          --use_projector $USE_PROJECTOR \
          --use_specific $USE_SPECIFIC \
          --use_transformer $USE_TRANSFORMER \
          --add_type $ADD_TYPE \
          --ce_only $CE_ONLY \
          --warmup $WARMUP \
          --cosine_warmup $COSINE_WARMUP \
          --alpha_sim $ALPHA_SIM \
          --alpha_diff $ALPHA_DIFF \
          --alpha_recon $ALPHA_RECON
        done
      done
    done
  done
done
echo "Finished."
