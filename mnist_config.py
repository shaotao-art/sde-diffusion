DEBUG = False
device = 'cuda'

num_ep=10
# optimizer
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # warm_up_epoch=1
    )
)


model_config = dict(
    sample_size=32,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 128, 256),  # the number of output channes for each UNet block
    norm_num_groups = 4,
    down_block_types=( 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
      )
)

# SDE 
sche_config = dict(
    beta_min=0.1,
    beta_max=20,
    num_infer_step=500
)

####---- data ----####
img_size = 32
dataset_type = 'mnist'
train_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root='.'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 2,
    )
)
val_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root='.'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 2,
    )
)
####---- data ----####


sample_every = 1000
num_valid_gen_sample = 16
resume_ckpt_path = None
# load_weight_from = None

# ckp
ckp_config = dict(
   save_last=True, 
   every_n_epochs=None
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    val_check_interval=1
)

# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'gen-models',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'