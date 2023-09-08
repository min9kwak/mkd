echo "Experiments Started"
SERVER=main
GPUS=0

N_TRAIN=1000

X1_DIM=50
X2_DIM=50
XS1_DIM=20
XS2_DIM=20
OVERLAP_DIM=10

ENCODER_ACT=relu

MM_MODE='increase_alpha'
MISSING_MODE='remove'
LINEAR_NETWORK=False

# increase_gamma: change overlap_dim
# increase_alpha: change xs1_dim (0, 5, 10, 15, 20)
# remove: n_train=500

for RANDOM_STATE in {2021..2050}; do
  for MISSING_RATE in 0.1 0.2 0.3 0.4 0.5 0.6; do
    for XS1_DIM in 0 5 10 15 20; do
      for XS2_DIM in 0 5 10 15 20; do
        python ./run_simulator.py \
        --server $SERVER \
        --gpus $GPUS \
        --n_train $N_TRAIN \
        --x1_dim $X1_DIM \
        --x2_dim $X2_DIM \
        --xs1_dim $XS1_DIM \
        --xs2_dim $XS2_DIM \
        --overlap_dim $OVERLAP_DIM \
        --missing_rate $MISSING_RATE \
        --encoder_act $ENCODER_ACT \
        --linear_network $LINEAR_NETWORK \
        --mm_mode=$MM_MODE \
        --missing_mode=$MISSING_MODE \
        --random_state $RANDOM_STATE
      done
    done
  done
done
echo "Finished."
