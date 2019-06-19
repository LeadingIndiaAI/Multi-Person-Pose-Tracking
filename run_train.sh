. venv/bin/activate

mpirun -np 4 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca btl_vader_single_copy_mechanism none \
    python3 train.py
