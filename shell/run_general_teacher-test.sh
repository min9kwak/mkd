echo "Experiments Started"
SERVER=main
GPUS=0

PET_TYPE=FBP
CROP_SIZE=64
MCI_ONLY=True
USE_UNLABELED=True
TRAIN_SLICES=fixed
SPACE=3
N_POINTS=5
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

ALPHA_SIM=1.0
ALPHA_DIFF=2.0
ALPHA_RECON=0.5

RANDOM_STATE=2021

USE_PROJECTOR=True
USE_SPECIFIC=False
USE_TRANSFORMER=False

AGG=sum
ADD_TYPE=add

BALANCE=True
SAMPLER_TYPE=stratified

for RANDOM_STATE in 2021
do
  for USE_SPECIFIC in False
  do
    for LEARNING_RATE in 0.001
    do
      for SPACE in 3
      do
        for SAMPLER_TYPE in stratified
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
          --space $SPACE \
          --n_points $N_POINTS \
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
          --balance $BALANCE \
          --sampler_type $SAMPLER_TYPE \
          --alpha_sim $ALPHA_SIM \
          --alpha_diff $ALPHA_DIFF \
          --alpha_recon $ALPHA_RECON \
          --agg $AGG
        done
      done
    done
  done
done
echo "Finished."
