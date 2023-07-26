echo "Experiments Started"
SERVER=workstation3
GPUS=00

DATA_TYPE=multi

PET_TYPE=FBP
MCI_ONLY=True
USE_UNLABELED=True
MISSING_RATE=-1

EXTENDED=True

EXTRACTOR_TYPE=resnet50
SMALL_KERNEL=True

EPOCHS=100
LEARNING_RATE=0.001

RANDOM_STATE=2021

ADD_TYPE=add

BALANCE=True
SAMPLER_TYPE=stratified

for RANDOM_STATE in {2021..2025}
do
	for LEARNING_RATE in 0.001
	do
	  for EXTRACTOR_TYPE in resnet50
	  do
      python ./run_multi.py \
      --gpus $GPUS \
      --server $SERVER \
      --data_type $DATA_TYPE \
      --pet_type $PET_TYPE \
      --mci_only $MCI_ONLY \
      --use_unlabeled $USE_UNLABELED \
      --data_file labels/data_info_multi.csv \
      --missing_rate $MISSING_RATE \
      --extractor_type $EXTRACTOR_TYPE \
      --small_kernel $SMALL_KERNEL \
      --random_state $RANDOM_STATE \
      --epochs $EPOCHS \
      --batch_size 16 \
      --optimizer adamw \
      --learning_rate $LEARNING_RATE \
      --weight_decay 0.0001 \
      --save_every 1000 \
      --balance $BALANCE \
      --sampler_type $SAMPLER_TYPE \
      --add_type $ADD_TYPE \
      --extended $EXTENDED
    done
	done
done
echo "Finished."
