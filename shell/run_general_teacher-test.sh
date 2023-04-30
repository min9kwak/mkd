echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
TRAIN_SLICES=fixed
MISSING_RATE=0.60

ENCODER_TYPE=resnet50
SMALL_KERNEL=True

HIDDEN=128
SWAP=False
MLP=False
DROPOUT=0.0

EPOCHS=30
LEARNING_RATE=0.001

RANDOM_STATE=2021

# for MLP, USE_PROJECTOR, USE_SPECIFIC, USE_TRANSFORMER, ADD_TYPE
CE_ONLY=True

for MLP in True False
do
  for USE_PROJECTOR in True False
  do
    for USE_SPECIFIC in True False
    do
      for USE_TRANSFORMER in True False
      do
        for ADD_TYPE in add concat
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
          --cosine_warmup -1
        done
      done
    done
  done
done
echo "Finished."

HIDDEN=64
for MLP in True False
do
  for USE_PROJECTOR in True False
  do
    for USE_SPECIFIC in True False
    do
      for USE_TRANSFORMER in True False
      do
        for ADD_TYPE in add concat
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
          --cosine_warmup -1
        done
      done
    done
  done
done
echo "Finished."
