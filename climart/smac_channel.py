import logging
from functools import partial
import os, sys, time, random, argparse, collections
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import numpy as np

from climart.data_wrangling.constants import TEST_YEARS, LAYERS, OOD_PRESENT_YEARS, TRAIN_YEARS, get_flux_mean, \
    get_data_dims, OOD_FUTURE_YEARS, OOD_HISTORIC_YEARS
from climart.data_wrangling.h5_dataset import ClimART_HdF5_Dataset
from climart.models.column_handler import ColumnPreprocesser
from climart.models.interface import get_trainer, is_gnn, is_graph_net, get_model, get_input_transform

from climart.utils.hyperparams_and_args import get_argparser
from climart.utils.preprocessing import Normalizer
from climart.utils.utils import set_seed, year_string_to_list, get_logger, get_target_variable, get_target_types

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.multi_objective.parego import ParEGO

torch.set_printoptions(sci_mode=False)
log = get_logger(__name__)

def get_cnn_config_space():
    cs = ConfigSpace.ConfigurationSpace()
    first_channel = CSH.CategoricalHyperparameter('first_channel', [100, 32, 50, 64], default_value=100)
    second_channel = CSH.CategoricalHyperparameter('second_channel', [200, 64, 100, 128], default_value=200)
    third_channel = CSH.CategoricalHyperparameter('third_channel', [400, 128, 200, 256], default_value=400)
    fourth_channel = CSH.CategoricalHyperparameter('fourth_channel', [100, 32, 50, 64], default_value=100)
    '''
    channel_list = CSH.CategoricalHyperparameter(
        name="channel_list",
        choices=[[100,200,400,100], [32,64,128,32], [50, 100, 200, 50], [100,200,400,100], [64, 128, 256, 64]],
        default_value=[100,200,400,100],
    )
    '''
    cs.add_hyperparameters([first_channel, second_channel, third_channel, fourth_channel])
    return cs

def train_cnn(cfg):
    net_params['channels_list'] = [cfg['first_channel'], cfg['second_channel'], cfg['third_channel'], 100]
    params['epochs'] = 5
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

    best_valid, eval_time = trainer.fit(trainloader, valloader,
                                hyper_params=params,
                                testloader=None,
                                testloader_names=None,
                                time_valid=True,
                                )

    return {'loss': float(best_valid), 'time': float(eval_time)}


if __name__ == '__main__':

    logging.basicConfig()
    global params
    global net_params
    global other_args
    params, net_params, other_args = get_argparser()
    set_seed(params['seed'])  # for reproducibility
    result = get_data_dims(params['exp_type'])
    spatial_dim = result['spatial_dim']
    in_dim = result['input_dim']

    global cp 
    if is_gnn(params['model']) or is_graph_net(params['model']):
        # cp maps the data to a graph structure needed for a GCN or GraphNet
        cp = ColumnPreprocesser(
            n_layers=spatial_dim[LAYERS], input_dims=in_dim, **params['preprocessing_dict']
        )
        input_transform = cp.get_preprocesser
    else:
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

    # Main Present-day Test Set(s):
    # To compute metrics for each test year, we will have a separate dataloader for each of the test years (2007-14).
    '''
    global test_names
    test_names = [f'Test_{test_year}' for test_year in TEST_YEARS]
    test_sets = [
        ClimART_HdF5_Dataset(years=[test_year], name=test_name, output_normalization=None, **dataset_kwargs)
        for test_year, test_name in zip(TEST_YEARS, test_names)
    ]
    '''
    net_params['input_dim'] = train_set.input_dim
    net_params['spatial_dim'] = train_set.spatial_dim
    net_params['out_dim'] = train_set.output_dim
    params['target_type'] = get_target_types(params.pop('target_type'))
    log.info(f" {'Targets are' if len(params['target_type']) > 1 else 'Target is'} {' '.join(params['target_type'])}")
    params['target_variable'] = get_target_variable(params.pop('target_variable'))
    params['training_set_size'] = len(train_set)
    global output_normalizer
    global output_postprocesser
    output_normalizer = train_set.output_normalizer
    output_postprocesser = train_set.output_variable_splitter

    global out_layer_bias
    if not isinstance(output_normalizer, Normalizer):
        log.info('Initializing out layer bias to output train dataset mean!')
        params['output_bias_mean_init'] = True
        out_layer_bias = get_flux_mean()
    else:
        params['output_bias_mean_init'] = False
        out_layer_bias = None

    dataloader_kwargs = {'pin_memory': True, 'num_workers': params['workers']}
    eval_batch_size = 512

    global trainloader
    global valloader
    trainloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, **dataloader_kwargs)
    valloader = DataLoader(val_set, batch_size=eval_batch_size, **dataloader_kwargs)

    '''
    global testloaders
    testloaders = [
        DataLoader(test_set, batch_size=eval_batch_size, **dataloader_kwargs) for test_set in test_sets
    ]
    '''

    cs = get_cnn_config_space()

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 50,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
            "multi_objectives": ["loss", "time"],
            # You can define individual crash costs for each objective
            "cost_for_crash": [1, float(MAXINT)],
        }
    )

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    # Pass the multi objective algorithm and its hyperparameters
    smac = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=train_cnn,
        multi_objective_algorithm=ParEGO,
        multi_objective_kwargs={
            "rho": 0.05,
        },
    )

    incumbent = smac.optimize()

    # pareto front based on smac.runhistory.data
    cost = np.vstack([v[0] for v in smac.runhistory.data.values()])


