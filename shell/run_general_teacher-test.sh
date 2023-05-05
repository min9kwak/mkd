echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
MCI_ONLY=True
USE_UNLABELED=True
TRAIN_SLICES=fixed
MISSING_RATE=-1

ENCODER_TYPE=resnet50
SMALL_KERNEL=True

HIDDEN=128
SWAP=False
MLP=False
DROPOUT=0.0
ENCODER_ACT=sigmoid

EPOCHS=100
LEARNING_RATE=0.001
COSINE_WARMUP=0

CE_ONLY=False
WARMUP=-1
LOSS_DIFF="mse"

ALPHA_SIM=0.1
ALPHA_DIFF=0.1
ALPHA_RECON=0.1

ALPHA=0.2

RANDOM_STATE=2021

USE_PROJECTOR=True
USE_SPECIFIC=True
USE_TRANSFORMER=False

AGG=sum

for RANDOM_STATE in 2021
do
  for USE_SPECIFIC in True
  do
    for AGG in sum
    do
      for ALPHA in 0.5
      do
        for ADD_TYPE in add
        do
          python ./run_general_teacher.py \
          --gpus $GPUS \
          --server $SERVER \
          --pet_type $PET_TYPE \
          --mci_only $MCI_ONLY \
          --use_unlabeled $USE_UNLABELED \
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
          --encoder_act $ENCODER_ACT \
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
          --alpha_sim $ALPHA \
          --alpha_diff $ALPHA \
          --alpha_recon $ALPHA \
          --agg $AGG
        done
      done
    done
  done
done
echo "Finished."
