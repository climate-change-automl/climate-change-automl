import logging
from functools import partial
import os, sys, time, random, argparse, collections
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from climart.data_wrangling.constants import TEST_YEARS, LAYERS, OOD_PRESENT_YEARS, TRAIN_YEARS, get_flux_mean, \
    get_data_dims, OOD_FUTURE_YEARS, OOD_HISTORIC_YEARS
from climart.data_wrangling.h5_dataset import ClimART_HdF5_Dataset
from climart.models.column_handler import ColumnPreprocesser
from climart.models.interface import get_trainer, is_gnn, is_graph_net, get_model, get_input_transform, is_cnn

from climart.utils.hyperparams_and_args import get_argparser
from climart.utils.preprocessing import Normalizer
from climart.utils.utils import set_seed, year_string_to_list, get_logger, get_target_variable, get_target_types

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB

from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.multi_objective.parego import ParEGO

torch.set_printoptions(sci_mode=False)
log = get_logger(__name__)

def is_pareto_efficient_simple(costs):
    """
    Plot the Pareto Front in our 2d example.

    source from: https://stackoverflow.com/a/40239615
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)

            # And keep self
            is_efficient[i] = True
    return is_efficient


def plot_pareto_from_runhistory(observations):
    """
    This is only an example function for 2d plotting, when both objectives
    are to be minimized
    """

    # find the pareto front
    efficient_mask = is_pareto_efficient_simple(observations)
    front = observations[efficient_mask]
    # observations = observations[np.invert(efficient_mask)]

    obs1, obs2 = observations[:, 0], observations[:, 1]
    front = front[front[:, 0].argsort()]

    # add the bounds
    x_upper = np.max(obs1)
    y_upper = np.max(obs2)
    front = np.vstack([[front[0][0], y_upper], front, [x_upper, np.min(front[:, 1])]])

    x_front, y_front = front[:, 0], front[:, 1]

    plt.scatter(obs1, obs2)
    plt.step(x_front, y_front, where="post", linestyle=":")
    plt.title("Pareto-Front")

    plt.xlabel("Cost")
    plt.ylabel("Time")
    plt.savefig("cost_pareto.jpg")
    
def get_network_config_space():
    cs = ConfigSpace.ConfigurationSpace()
    model_choice = CSH.CategoricalHyperparameter('network_type', ['LGCN+Readout', 'CNN', 'MLP', 'GCN+Readout', 'GN+Readout'], default_value='MLP')
    #model_choice = CSH.CategoricalHyperparameter('network_type', ['CNN', 'MLP'], default_value='MLP')
    lr = CSH.UniformFloatHyperparameter("lr", lower=1e-5, upper=1e-1, default_value=2e-4, log=True)
    wd = CSH.UniformFloatHyperparameter("weight_decay", lower=1e-7, upper=1e-4, default_value=1e-6, log=True)

    cs.add_hyperparameters([model_choice, lr, wd])
    return cs

def train_network(cfg):
    params['model'] = cfg['network_type']
    params['lr'] = float(cfg['lr'])
    params['weight_decay'] = float(cfg['weight_decay'])

    params['epochs'] = 5
    net_params['channels_list'] = [100, 200, 400, 100]

    if params['model'].strip().lower() == 'mlp':
        net_params['hidden_dims'] = [512, 256, 256]
        net_params['net_normalization'] = 'layer_norm'
        params['preprocessing_dict']['preprocessing'] = "padding"


    if params['model'] == 'LGCN+Readout':
        params['model'] = 'GCN+Readout'
        net_params['learn_edge_structure'] = True
    else: 
        net_params['learn_edge_structure'] = False

    if is_gnn(params['model']):
        net_params['hidden_dims'] = [128, 128, 128]
        net_params['net_normalization'] = 'layer_norm'
        params['preprocessing_dict']['preprocessing'] = "mlp_projection"

    
    if is_graph_net(params['model']):
        net_params['hidden_dims'] = [128, 128, 128]
        net_params['net_normalization'] = 'layer_norm'
        params['preprocessing_dict']['preprocessing'] = "graph_net_level_nodes"
    
    if is_cnn(params['model']):
        net_params['hidden_dims'] = [128, 128, 128]
        net_params['net_normalization'] = 'none'
        params['preprocessing_dict']['preprocessing'] = "padding"

    

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

    '''
    global testloaders
    testloaders = [
        DataLoader(test_set, batch_size=eval_batch_size, **dataloader_kwargs) for test_set in test_sets
    ]
    '''
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

    cs = get_network_config_space()

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 30,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
            "multi_objectives": ["loss", "time"],
            # You can define individual crash costs for each objective
            "cost_for_crash": [1000, float(MAXINT)],
        }
    )

    # Optimize, using a SMAC-object
    # Pass the multi objective algorithm and its hyperparameters
    smac = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=train_network,
        multi_objective_algorithm=ParEGO,
        multi_objective_kwargs={
            "rho": 0.05,
        },
    )

    incumbent = smac.optimize()

    # pareto front based on smac.runhistory.data
    cost = np.vstack([v[0] for v in smac.runhistory.data.values()])
    plot_pareto_from_runhistory(cost)



