import logging
from functools import partial
import os, sys, time, random, argparse, collections
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import numpy as np
import optuna 

from climart.data_wrangling.constants import TEST_YEARS, LAYERS, OOD_PRESENT_YEARS, TRAIN_YEARS, get_flux_mean, \
    get_data_dims, OOD_FUTURE_YEARS, OOD_HISTORIC_YEARS
from climart.data_wrangling.h5_dataset import ClimART_HdF5_Dataset
from climart.models.column_handler import ColumnPreprocesser
from climart.models.interface import get_trainer, is_gnn, is_graph_net, get_model, get_input_transform

from climart.utils.hyperparams_and_args import get_argparser
from climart.utils.preprocessing import Normalizer
from climart.utils.utils import set_seed, year_string_to_list, get_logger, get_target_variable, get_target_types

torch.set_printoptions(sci_mode=False)
log = get_logger(__name__)
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    bs = trial.suggest_float('batch_size', 7, 9)

    params['lr'] = lr
    params['weight_decay'] = wd
    net_params['dropout'] = dropout 
    params['batch_size'] = int(2**int(bs))

    cp = None
    input_transform = partial(get_input_transform, model_class=get_model(params['model'], only_class=True))
    dataset_kwargs = dict(
        exp_type=params['exp_type'],
        target_type=params['target_type'],
        target_variable=params['target_variable'],
        input_transform=input_transform,
        input_normalization=params['in_normalize'],
        spatial_normalization_in=params['spatial_normalization_in'],
        log_scaling=params['log_scaling'],
    )
    # Training set:
    train_years = year_string_to_list(params['train_years'])
    assert all([y in TRAIN_YEARS for y in train_years]), f"All years in --train_years must be in {TRAIN_YEARS}!"
    train_set = ClimART_HdF5_Dataset(years=train_years, name='Train',
                                    output_normalization=params['out_normalize'],
                                    spatial_normalization_out=params['spatial_normalization_out'],
                                    load_h5_into_mem=params['load_train_into_mem'],
                                    **dataset_kwargs)
    # Validation set:
    val_set = ClimART_HdF5_Dataset(years=year_string_to_list(params['validation_years']), name='Val',
                                output_normalization=None,
                                load_h5_into_mem=params['load_val_into_mem'],
                                **dataset_kwargs)


    net_params['input_dim'] = train_set.input_dim
    net_params['spatial_dim'] = train_set.spatial_dim
    net_params['out_dim'] = train_set.output_dim
    params['target_type'] = get_target_types(params.pop('target_type'))
    log.info(f" {'Targets are' if len(params['target_type']) > 1 else 'Target is'} {' '.join(params['target_type'])}")
    params['target_variable'] = get_target_variable(params.pop('target_variable'))
    params['training_set_size'] = len(train_set)

    output_normalizer = train_set.output_normalizer
    output_postprocesser = train_set.output_variable_splitter

    if not isinstance(output_normalizer, Normalizer):
        log.info('Initializing out layer bias to output train dataset mean!')
        params['output_bias_mean_init'] = True
        out_layer_bias = get_flux_mean()
    else:
        params['output_bias_mean_init'] = False
        out_layer_bias = None

    dataloader_kwargs = {'pin_memory': True, 'num_workers': params['workers']}
    eval_batch_size = 512

    trainloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, **dataloader_kwargs)
    valloader = DataLoader(val_set, batch_size=eval_batch_size, **dataloader_kwargs)

    trainer_kwargs = dict(
        model_name=params['model'], model_params=net_params,
        device=params['device'], seed=params['seed'],
        model_dir=params['model_dir'],
        out_layer_bias=out_layer_bias,
        output_postprocesser=output_postprocesser,
        output_normalizer=output_normalizer,
    )
    if cp is not None:
        trainer_kwargs['column_preprocesser'] = cp

    trainer = get_trainer(**trainer_kwargs)

    best_valid = trainer.fit(trainloader, valloader,
                                hyper_params=params,
                                testloader=None,
                                testloader_names=None,
                                )

    return float(best_valid)


if __name__ == '__main__':
    logging.basicConfig()
    global params
    global net_params
    global other_args
    params, net_params, other_args = get_argparser()
    set_seed(params['seed'])  # for reproducibility

    study_name = "climart-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', load_if_exists=True)
    #for smaller budget, run 24 trials, else 32
    study.optimize(objective, n_trials=32)

    best_lr = study.best_params['lr']
    print('best found learning rate is: ')
    print(best_lr)

    best_wd = study.best_params['weight_decay']
    print('best found wd is: ')
    print(best_wd)

    best_dropout = study.best_params['dropout']
    print('best found dropout rate is ')
    print(best_dropout)

    best_bs = study.best_params['batch_size']
    print('best found batch size is ')
    print(int(2**int(best_bs)))

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
