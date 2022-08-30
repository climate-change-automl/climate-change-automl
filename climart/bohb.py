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
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

torch.set_printoptions(sci_mode=False)
log = get_logger(__name__)

def get_cnn_config_space():
    cs = ConfigSpace.ConfigurationSpace()
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value='1e-3', log=True)
    cs.add_hyperparameters([lr])
    return cs

class MyWorker(Worker):
    def __init__(self, params, net_params, other_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_time = []
        self.trajectory = []

        result = get_data_dims(params['exp_type'])
        spatial_dim = result['spatial_dim']
        in_dim = result['input_dim']
        if is_gnn(params['model']) or is_graph_net(params['model']):
            # cp maps the data to a graph structure needed for a GCN or GraphNet
            self.cp = ColumnPreprocesser(
                n_layers=spatial_dim[LAYERS], input_dims=in_dim, **params['preprocessing_dict']
            )
            input_transform = cp.get_preprocesser
        else:
            self.cp = None
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
        self.test_names = [f'Test_{test_year}' for test_year in TEST_YEARS]
        test_sets = [
            ClimART_HdF5_Dataset(years=[test_year], name=test_name, output_normalization=None, **dataset_kwargs)
            for test_year, test_name in zip(TEST_YEARS, self.test_names)
        ]

        net_params['input_dim'] = train_set.input_dim
        net_params['spatial_dim'] = train_set.spatial_dim
        net_params['out_dim'] = train_set.output_dim
        params['target_type'] = get_target_types(params.pop('target_type'))
        log.info(f" {'Targets are' if len(params['target_type']) > 1 else 'Target is'} {' '.join(params['target_type'])}")
        params['target_variable'] = get_target_variable(params.pop('target_variable'))
        params['training_set_size'] = len(train_set)
        self.output_normalizer = train_set.output_normalizer
        self.output_postprocesser = train_set.output_variable_splitter

        if not isinstance(self.output_normalizer, Normalizer):
            log.info('Initializing out layer bias to output train dataset mean!')
            params['output_bias_mean_init'] = True
            self.out_layer_bias = get_flux_mean()
        else:
            params['output_bias_mean_init'] = False
            self.out_layer_bias = None

        dataloader_kwargs = {'pin_memory': True, 'num_workers': params['workers']}
        eval_batch_size = 512
        self.trainloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, **dataloader_kwargs)
        self.valloader = DataLoader(val_set, batch_size=eval_batch_size, **dataloader_kwargs)

        self.testloaders = [
            DataLoader(test_set, batch_size=eval_batch_size, **dataloader_kwargs) for test_set in test_sets
        ]
        self.params = params
        self.net_params = net_params 
        self.other_args = other_args
    
    def compute(self, config, budget, *args, **kwargs):
        self.params['epochs'] = int(budget)
        self.params['lr'] = config['lr']
        
        results = self.learn(self.params, self.net_params, self.other_args)
        self.trajectory.append([results['loss'], config['lr']])
        return results


    def learn(self, params, net_params, other_args, only_final_eval=False, *args, **kwargs):

        trainer_kwargs = dict(
            model_name=params['model'], model_params=net_params,
            device=params['device'], seed=params['seed'],
            model_dir=params['model_dir'],
            out_layer_bias=self.out_layer_bias,
            output_postprocesser=self.output_postprocesser,
            output_normalizer=self.output_normalizer,
        )
        if self.cp is not None:
            trainer_kwargs['column_preprocesser'] = self.cp

        trainer = get_trainer(**trainer_kwargs)

        best_valid = trainer.fit(self.trainloader, self.valloader,
                                    hyper_params=params,
                                    testloader=self.testloaders,
                                    testloader_names=self.test_names,
                                    *args, **kwargs)
        log.info(f" Testing the best model as measured by validation performance (best={best_valid:.3f})")


        #return ({'loss': best_valid, 'info': {'test_stats': final_test_stats}})
        return {'loss': float(best_valid)}


if __name__ == '__main__':
    logging.basicConfig()
    params, net_params, other_args = get_argparser()
    set_seed(params['seed'])  # for reproducibility
    cs = get_cnn_config_space()

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            params, 
            net_params,
            other_args,
            nameserver=ns_host,
            nameserver_port=ns_port,
            run_id=hb_run_id,
        )
        w.run(background=True)
        workers.append(w)

    start_time = time.time()


    bohb = BOHB(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=12,
        nameserver=ns_host,
        nameserver_port=ns_port,
        num_samples=64,
        random_fraction=0.33,
        bandwidth_factor=3,
        ping_interval=50,
        min_bandwidth=0.3,
    )

    results = bohb.run(1, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    current_best_lr = []
    for idx in range(len(workers[0].trajectory)):
        trajectory = workers[0].trajectory[: idx + 1]
        lr = min(trajectory, key=lambda x: x[0])[1]
        current_best_lr.append(lr)
    
    best_lr = min(workers[0].trajectory, key=lambda x: x[0])[1]
    log.info(f'best lr is {best_lr}.')



