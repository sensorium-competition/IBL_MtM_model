import os

from experanto.dataloaders import get_multisession_dataloader
from experanto.configs import DEFAULT_CONFIG as cfg


def build_dataloader(config, inference=False):

    # Data paths.
    path_pre = config['data']['path']
    full_paths = [os.path.join(path_pre, f) for f in config['data']['scans']]

    # Overwrite default config with custom values.
    # Overwrite default config with custom values.
    if 'screen_chunk_size' in config['data']:
        scale = config['data']['response_chunk_size'] / config['data']['response_sampling_rate']
        cfg.dataset.modality_config.screen.chunk_size = config['data'][
            'screen_chunk_size'
        ] if config['data']['screen_chunk_size'] is not None else int(config['data']['screen_sampling_rate'] * scale)

        cfg.dataset.modality_config.screen.sampling_rate = config['data'][
            'screen_sampling_rate'
        ]
        cfg.dataset.modality_config.eye_tracker.chunk_size = config['data'][
            'eye_tracker_chunk_size'
        ] if config['data']['eye_tracker_chunk_size'] is not None else int(config['data']['eye_tracker_sampling_rate'] * scale)

        cfg.dataset.modality_config.eye_tracker.sampling_rate = config['data'][
            'eye_tracker_sampling_rate'
        ]
        cfg.dataset.modality_config.treadmill.chunk_size = config['data'][
            'treadmill_chunk_size'
        ] if config['data']['treadmill_chunk_size'] is not None else int(config['data']['treadmill_sampling_rate'] * scale)
        cfg.dataset.modality_config.treadmill.sampling_rate = config['data'][
            'treadmill_sampling_rate'
        ]

        cfg.dataset.modality_config.responses.chunk_size = config['data'][
            'response_chunk_size'
        ]
        cfg.dataset.modality_config.responses.sampling_rate = config['data'][
            'response_sampling_rate'
        ]

    else:
        cfg.dataset.global_chunk_size = config['data']['chunk_size']
        cfg.dataset.global_sampling_rate = config['data']['sampling_rate']
    
    cfg.dataset.modality_config.screen.include_blanks = False
    cfg.dataset.modality_config.screen.transforms.Resize.size = config['data']['img_size']

    cfg.dataset.modality_config.responses.offset =  config['data']['responses_offset']

    
    if 'sample_stride' in config['data']:
        cfg.dataset.modality_config.screen.sample_stride = config['data']['sample_stride']
    elif inference:
        cfg.dataset.modality_config.screen.sample_stride = (
            config['data']['screen_chunk_size']
            if 'screen_chunk_size' in config['data']
            else config['data']['chunk_size']
        )
    else:
        cfg.dataset.modality_config.screen.sample_stride = 1

    cfg.dataloader.num_workers = config['data']['num_workers']
    cfg.dataloader.prefetch_factor = 2
    cfg.dataloader.batch_size = config['data']['batch_size']
    cfg.dataloader.pin_memory = True

    # Training datasets.
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg.dataloader.shuffle = True
    train_loader = get_multisession_dataloader(full_paths, cfg)

    # Validation datasets.
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "oracle"}
    cfg.dataloader.shuffle = False
    # todo - validation without overlaps
    cfg.dataset.modality_config.screen.sample_stride = (
            config['data']['screen_chunk_size']
            if 'screen_chunk_size' in config['data']
            else config['data']['chunk_size']
        )
    val_loader = get_multisession_dataloader(full_paths, cfg)

    return train_loader, val_loader
