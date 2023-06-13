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
MLP=False
DROPOUT=0.0

EPOCHS=100
LEARNING_RATE=0.0001

ALPHA_SIM=10.0
ALPHA_DIFF=10.0
ALPHA_RECON=10.0


for RANDOM_STATE in 2021 2022 2023
do
  for ALPHA_SIM in 1.0 2.0 3.0
  do
    for LEARNING_RATE in 0.0001 0.001
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
      --alpha_sim $ALPHA_SIM \
      --alpha_diff $ALPHA_DIFF \
      --alpha_recon $ALPHA_RECON
    done
  done
done
echo "Finished."
