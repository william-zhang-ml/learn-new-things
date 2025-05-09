BLUF: gradient checkpointing with mixed precision forward passes let me triple
max batchsize.

Ad-hoc noodling around with training code for an old, small GPU.
- RTX3080 - about 10GB GPU memory, 1GB used for doing normal computer things.
- CIFAR10 (resized to 224x224)
- ResNet50
- `torch=1.12.1`

According to my mentor, small GPU training meta involves 3 things that more or less let you get larger batches.
- automatic mixed precision
- gradient accumulation
- gradient checkpoints

Scripts
- Baseline -> batchsize of 64, 2-3 minutes per epoch
- Grad Accum -> accumulate gradient N times for an effective batchsize of 64N, 2-3 minutes per epoch
- Amp -> automatic mixed precision (w/o gradient scaling), batchsize of 128, 1-2 minutes per epoch
- Grad Ckpt -> gradient checkpoints (do not store intermediate tensors but have to double inference), batchsize of 96, 4-5 minutes per epoch
- All 3 -> batchsize of 192, 2-3 minutes per epoch

Notes:
- was wondering if the first epoch is slower than successive epochs, answer seems to be no
- AMP results in smaller intermediate tensors which lets you increase batchsize
- grad checkpoints need you to set `requires_grad = True` on the input
    - this will make code angry at inplace operations
    - in theory this is okay if you double-forward, but unsure how to make that happen
- grad checkpoints are minmax-ing
    - generally want to checkpoint small tensors to keep more memory open
    - but larger tensors are probably slow to compute and we don't want to compute them twice
    - number of segments is also a factor since each segment will use memory to store its checkpoint
    - if done poorly, this (true bigger batch) can be slower than gradient accumulation (synthetic bigger batch)
- using AMP and grad checkpoints together is a game changer because they compensate for each other's weakness
    - grad checkpoints only means double inference but AMP makes inference faster