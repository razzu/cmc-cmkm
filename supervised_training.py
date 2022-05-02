import argparse

from pytorch_lightning import Trainer, seed_everything

from utils.experiment_utils import generate_experiment_id, load_yaml_to_dict
from utils.training_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    
    # Configs
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--tuning_config_path', default=None)

    # Data and models
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_save_path', default='./model_weights')

    parser.add_argument('--no_ckpt', action='store_true', default=False)
    
    return parser.parse_args()


def train_test_supervised_model(args, cfg, dataset_cfg, freeze_encoder=False, approach='supervised', experiment_info=None, limited_k=None):
    experiment_id = generate_experiment_id()

    modality = list(cfg['modalities'].keys())[0] # assume unimodal for now
    batch_size = cfg['modalities'][modality]['model'][args.model]['kwargs']['batch_size']
    num_epochs = cfg['experiment']['num_epochs']

    model_cfg = cfg['modalities'][modality]['model'][args.model]
    transform_cfg = cfg['modalities'][modality]['transforms']
    model_cfg, transform_cfg = check_sampling_cfg(model_cfg, transform_cfg)
    train_transforms, test_transforms = init_transforms(modality, transform_cfg)
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=[modality], batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
        limited_k=limited_k)

    # Merge general model params with dataset-specific model params.
    model_cfg['kwargs'] = {**dataset_cfg[modality], **model_cfg['kwargs']}
    model = init_model(model_cfg, dataset_cfg['main_metric'])

    if freeze_encoder:
        getattr(model, model_cfg['encoder_name']).freeze()

    if experiment_info is None:
        experiment_info = {
            "dataset": args.dataset,
            "model": model_cfg['class_name']
        }

    callbacks = setup_callbacks(
        early_stopping_metric = "val_accuracy",
        early_stopping_mode   = "max",
        class_names           = dataset_cfg["class_names"],
        num_classes           = len(dataset_cfg["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg['main_metric'], 
        dataset               = args.dataset, 
        model                 = args.model, 
        experiment_id         = experiment_id
    )
    # setup loggers: tensorboards and/or wandb
    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality=modality, dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, approach=approach)

    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0, callbacks=callbacks, checkpoint_callback=not args.no_ckpt)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}

    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics


def main():
    args = parse_arguments()
    cfg = load_yaml_to_dict(args.experiment_config_path)
    seed_everything(cfg['experiment']['seed'])
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    modality = list(cfg['modalities'].keys())[0] 
    if args.tuning_config_path is None:
        train_test_supervised_model(args, cfg, dataset_cfg)
    else:
        tuning_cfg_combinations = get_tuning_grid_list(args.tuning_config_path, modality, args.model)
        for combination in tuning_cfg_combinations:
            print(combination)
            cfg['modalities'][modality]['model'][args.model]['kwargs'] = {**cfg['modalities'][modality]['model'][args.model]['kwargs'], **combination}
            train_test_supervised_model(args, cfg, dataset_cfg)


if __name__ == '__main__':
    main()
