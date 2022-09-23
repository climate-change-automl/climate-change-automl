from data_processed import *
from prepare import prep_env
from models import BaselineGruModel
from torch.utils.data import DataLoader
from dataset import TrainDataset
import torch
from torch import nn
import time
import os
import random
import sys

import optuna
from optuna.trial import TrialState


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def objective(trial):
    settings = prep_env(trial)
    continuous_training = len(sys.argv) > 1 and sys.argv[1] == 'continuous_training'
    if continuous_training:  # only finetune the first 36 output
        settings['input_len'] = 36
        settings['output_len'] = 36
        settings['epoch_num'] = 10

    model_name = f"model_{settings['seq_pre']}"

    seed_everything()  # ?

    training_data_list, col_num, _, _ = get_train_data_part(settings,trial)
    settings["in_var"] = col_num

 ################### OPTUNA HYPERPARAMS ##################
    hyperparams = {
        'num_sz' : trial.suggest_int("n_num_sz",32,64),
        'time_sz' : trial.suggest_int("n_time_sz",4,8),
        'id_sz' : trial.suggest_int("n_id_sz",4,8),
        'hl' : trial.suggest_int("hl",32,64)
    }
 ################### OPTUNA HYPERPARAMS ##################

    total_loss = 0

    for i in range(settings['part_num']):
        seed_everything(settings['random_seed'])
        train_features, train_features_cat, train_targets = training_data_list[i]
        train_dataset = TrainDataset(train_features, train_features_cat, train_targets, settings)
        train_dataloader = DataLoader(train_dataset, batch_size=settings['batch_size'], shuffle=True, num_workers=4)
        del train_features

        model = BaselineGruModel(settings,hyperparams).cuda()
        if continuous_training:  # load pretrain model
            path_to_pretrain_model = settings['checkpoints'] + f'/gru/o288/{model_name}_{i}.pth'
            model.load_state_dict(torch.load(path_to_pretrain_model))
            print('load', path_to_pretrain_model, 'successfully!')

        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        criterion1 = nn.MSELoss(reduction='none')
        steps_per_epoch = len(train_dataloader)

        print(f">>>>>>> Training Turbine   {i} >>>>>>>>>>>>>>>>>>>>>>>>>>")
        final_loss = 0
        for epoch in range(settings['epoch_num']):
            model.train()
            train_loss = 0
            t = time.time()
            for step, batch in enumerate(train_dataloader):
                features, featues_cat, targets = batch
                features = features.cuda()
                featues_cat = featues_cat.cuda()
                targets = targets[:, -settings['output_len']:].cuda()
                optimizer.zero_grad()
                output = model(features, featues_cat, settings['output_len'])
                loss = criterion1(output, targets)
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                print("epoch {}, Step [{}/{}] Loss: {:.3f} Time: {:.1f}s"
                      .format(epoch + 1, step + 1, steps_per_epoch, train_loss / (step + 1), time.time() - t),
                      end='\r', flush=True)
            print('')
            final_loss = train_loss

        if continuous_training:
            torch.save(model.state_dict(), settings['checkpoints'] + f'/gru/o36/{model_name}_{i}.pth')
        else:
            torch.save(model.state_dict(), settings['checkpoints'] + f'/gru/o288/{model_name}_{i}.pth')
        
        total_loss = (total_loss*(i) + final_loss )/ (i+1)
        trial.report(total_loss, i)

        if(trial.should_prune()):
            raise optuna.exceptions.TrialPruned()

    return total_loss

if __name__ == "__main__":
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 20)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key,value in trial.params.items():
        print(" {}: {}".format(key,value))

