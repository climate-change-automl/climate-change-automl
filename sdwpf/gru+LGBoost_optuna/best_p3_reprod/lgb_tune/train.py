
from feature_engineering import *
from prepare import prep_env
import lightgbm as lgb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import optuna
import sklearn.metrics
def objective(trial, index, part_num, x_train, x_val, y_train, y_val):
    if index == 288:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            "device": "gpu",
            'num_leaves': trial.suggest_int("num_leaves",2,128),
            "random_state": 2022,
            "bagging_freq": trial.suggest_int("bagging freq",1,7),
            "bagging_fraction": trial.suggest_float("bagging_frac",0.4,1),
            "feature_fraction": trial.suggest_float("feature_frac",0.4,1),
        }
    elif index == 72:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            "device": "gpu",
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            'num_leaves': trial.suggest_int("num_leaves",2,128),
            "random_state": 2022,
            "bagging_freq": trial.suggest_int("bagging freq",1,7),
            "bagging_fraction": trial.suggest_float("bagging_frac",0.4,1),
            "feature_fraction": trial.suggest_float("feature_frac",0.4,1),
        }
    elif index == 36:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            "device": "gpu",
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            'num_leaves': trial.suggest_int("num_leaves",2,128),
            "random_state": 2022,
            "bagging_freq": trial.suggest_int("bagging freq",1,7),
            "bagging_fraction": trial.suggest_float("bagging_frac",0.4,1),
            "feature_fraction": trial.suggest_float("feature_frac",0.4,1),
        }
    elif index == 18:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            "device": "gpu",
            'num_leaves': trial.suggest_int("num_leaves",2,128),
            "random_state": 2022,
            "bagging_freq": trial.suggest_int("bagging freq",1,7),
            "bagging_fraction": trial.suggest_float("bagging_frac",0.4,1),
            "feature_fraction": trial.suggest_float("feature_frac",0.4,1),
        }
    elif index == 9:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            "device": "gpu",
            'num_leaves': trial.suggest_int("num_leaves",2,128),
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            "random_state": 2022,
            "bagging_freq": trial.suggest_int("bagging freq",1,7),
            "bagging_fraction": trial.suggest_float("bagging_frac",0.4,1),
            "feature_fraction": trial.suggest_float("feature_frac",0.4,1),
        }
    else:
        params = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            'learning_rate': trial.suggest_float("lr",0.001,0.7),
            "device": "gpu",
        }

    model_name = "model_" + str(part_num) + "_" + str(index)
    label_name = 'target' + str(index)
    print(f"------------------train  {model_name}---------------------------")
    train_data = lgb.Dataset(x_train, label=y_train[label_name])
    valid_data = lgb.Dataset(x_val, label=y_val[label_name])
    gbm = lgb.train(params,
                    train_data,
                    valid_sets=[train_data, valid_data],
                    num_boost_round=1000,
                    verbose_eval=50,
                    early_stopping_rounds=20,
                    keep_training_booster=True
                    )
    eval = gbm.eval_valid()
    rmse = eval[0][2]
    #gbm.save_model(path_to_model + '/tree/' + model_name)
    return rmse

if __name__ == "__main__":
    settings = prep_env()
    path_to_model = settings["checkpoints"]
    df = get_train_data(settings,from_cache=False)
    df = add_target(df,settings)
    for part_num in range(4):
        x_train, x_val, y_train, y_val = split_data_by_part(df, settings,part_num+1,34)
        index = 0
        for i in settings['split_part']:
            index += i
            func = lambda trial: objective(trial,index,part_num,x_train,x_val,y_train,y_val)
            study = optuna.create_study(direction = "minimize")
            study.optimize(func, n_trials = 50)
            model_name = "model_" + str(part_num) + "_" + str(index)
            print("Best trial for 000111222 " + str(model_name))
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))


