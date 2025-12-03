from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    load_dataset_builder,
)
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.stpatch import STPatch
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer
import threading
from loader.dataset import build_dataloader
import json
import huggingface_hub
import wandb

# huggingface_hub.login(<Enter your hugging face token here>)
def train():
    # load config
    kwargs = {"model": "include:src/configs/ndt1_stitching_prompting.yaml"}

    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/ndt1_stitching_prompting.yaml", config)
    # config = update_config("src/configs/ssl_sessions_trainer.yaml", config)
    config = update_config("src/configs/ssl_sessions_trainer_8mice_stride_60.yaml", config)

    # set seed for reproducibility
    set_seed(config.seed)

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name
    )

    log_dir = os.path.join(
        config.dirs.log_dir,
        wandb.run.name.replace('-', '_')
        # config.wandb.run_name.replace('-', '_')
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config.wandb.run_name = wandb.run.name

    # Get hyperparameters from wandb sweep
    sweep_config = wandb.config
    
    # Override config with wandb sweep parameters
    # SSL sessions trainer parameters
    if 'use_lograte' in sweep_config:
        config['method']['model_kwargs']['use_lograte'] = sweep_config['use_lograte']
    
    if 'lr' in sweep_config:
        config['optimizer']['lr'] = sweep_config['lr']
    
    if 'wd' in sweep_config:
        config['optimizer']['wd'] = sweep_config['wd']
    
    if 'train_batch_size' in sweep_config:
        config['training']['train_batch_size'] = sweep_config['train_batch_size']
    
    # NDT1 stitching prompting parameters
    if 'masking_ratio' in sweep_config:
        config['masking_ratio'] = sweep_config['masking_ratio']
        config['model']['encoder']['masker']['ratio'] = sweep_config['masking_ratio']
    
    # Transformer parameters
    print(config.keys())
    print('\n\n')
    print(config)
    if 'n_layers' in sweep_config:
        config['model']['encoder']['transformer']['n_layers'] = sweep_config['n_layers']
    
    if 'hidden_size' in sweep_config:
        config['model']['encoder']['transformer']['hidden_size'] = sweep_config['hidden_size']
    
    if 'n_heads' in sweep_config:
        config['model']['encoder']['transformer']['n_heads'] = sweep_config['n_heads']
    
    if 'inter_size' in sweep_config:
        config['model']['encoder']['transformer']['inter_size'] = sweep_config['inter_size']
    
    if 'transformer_dropout' in sweep_config:
        config['model']['encoder']['transformer']['dropout'] = sweep_config['transformer_dropout']
    
    # Embedder parameters
    if 'n_channels' in sweep_config:
        config['model']['encoder']['embedder']['n_channels'] = sweep_config['n_channels']
    
    if 'mult' in sweep_config:
        config['model']['encoder']['embedder']['mult'] = sweep_config['mult']
    
    if 'embedder_dropout' in sweep_config:
        config['model']['encoder']['embedder']['dropout'] = sweep_config['embedder_dropout']
    
    # Update wandb config with the modified config
    wandb.config.update(config, allow_val_change=True)


    # download dataset from huggingface
    eid = None
    with open('/user/turishcheva/u14642/IBL_MtM_model/src/configs/config_stride_60_8_mice.json', 'r') as file:
        loader_config = json.load(file)

    print('Create Dataloader.')
    loader_config['data']['batch_size'] = config.training.train_batch_size
    train_dataloader, val_dataloader = build_dataloader(loader_config)
    print('Dataloader Created')

    meta_data = {"num_neurons": [], "num_sessions": 0, "eids": []}
    for key, v in train_dataloader.loaders.items():
        meta_data["num_neurons"].append(next(iter(v))['responses'].shape[-1])
        meta_data["num_sessions"] += 1
        meta_data["eids"].append(key)

    print(f'list(train_dataloader.loaders.keys()) = {list(train_dataloader.loaders.keys())}')
    print(f'meta_data={meta_data}')
    num_sessions = len(meta_data["eids"])

    # # wandb
    # if config.wandb.use:
    #     wandb.init(
    #         project=config.wandb.project,
    #         entity=config.wandb.entity,
    #         config=config,
    #         name=config.wandb.run_name
    #         # name="train_model_{}_num_session_{}_method_{}_mask_{}_stitch_{}".format(
    #         #     config.model.model_class,
    #         #     num_sessions,
    #         #     config.method.model_kwargs.method_name,
    #         #     config.encoder.masker.mode,
    #         #     config.encoder.stitching,
    #         # ),
    #     )

    # # make the dataloader
    # train_dataloader = make_loader(
    #     train_dataset,
    #     target=config.data.target,
    #     load_meta=config.data.load_meta,
    #     batch_size=config.training.train_batch_size,
    #     pad_to_right=True,
    #     pad_value=-1.0,
    #     max_time_length=config.data.max_time_length,
    #     max_space_length=config.data.max_space_length,
    #     dataset_name=config.data.dataset_name,
    #     sort_by_depth=config.data.sort_by_depth,
    #     sort_by_region=config.data.sort_by_region,
    #     stitching=config.encoder.stitching,
    #     shuffle=True,
    # )
    # # /mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/IBL_MtM_model/src/loader/base.py _preprocess_ibl_dataset
    # # return {
    # #     "spikes_data": binned_spikes_data,
    # #     "time_attn_mask": time_attn_mask,
    # #     "space_attn_mask": space_attn_mask,
    # #     "spikes_timestamps": spikes_timestamps,
    # #     "spikes_spacestamps": spikes_spacestamps,
    # #     "target": target_behavior,
    # #     "neuron_depths": neuron_depths, 
    # #     "neuron_regions": list(neuron_regions),
    # #     "eid": data['eid']
    # # }

    # val_dataloader = make_loader(
    #     val_dataset,
    #     target=config.data.target,
    #     load_meta=config.data.load_meta,
    #     batch_size=config.training.test_batch_size,
    #     pad_to_right=True,
    #     pad_value=-1.0,
    #     max_time_length=config.data.max_time_length,
    #     max_space_length=config.data.max_space_length,
    #     dataset_name=config.data.dataset_name,
    #     sort_by_depth=config.data.sort_by_depth,
    #     sort_by_region=config.data.sort_by_region,
    #     stitching=config.encoder.stitching,
    #     shuffle=False,
    # )

    # make log dir
    # log_dir = os.path.join(
    #     config.dirs.log_dir,
    #     wandb.run.name.replace('-', '_')
    #     # config.wandb.run_name.replace('-', '_')
    # )
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # config.wandb.run_name = wandb.run.name
    # Initialize the accelerator
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")
    # load model
    NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}

    config = update_config(config, meta_data)
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs, **meta_data)
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps,
    )
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        total_steps=config.training.num_epochs * len(train_dataloader) // config.optimizer.gradient_accumulation_steps,
        max_lr=config.optimizer.lr,
        pct_start=config.optimizer.warmup_pct,
        div_factor=config.optimizer.div_factor,
    )

    trainer_kwargs = {
        "log_dir": log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "config": config,
        "stitching": config.encoder.stitching,
    }
    trainer = make_trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        **trainer_kwargs,
        **meta_data
    )
    # Shared variable to signal the dummy load to stop
    stop_dummy_load = threading.Event()
    if config.training.dummy:
        # This is for HPC GPU usage, to avoid the GPU being idle
        print("Running dummy load")
        # Run dummy load in a separate thread
        dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load,))
        dummy_thread.start()
        try:
            # train loop
            trainer.train()
        finally:
            # Signal the dummy load to stop and wait for the thread to finish
            stop_dummy_load.set()
            dummy_thread.join()
    else:
        # train loop
        trainer.train()

if __name__ == "__main__":
    train()