# README_is2res

Download the data.
``` bash
$ cd examples/oc20/ && mkdir data && cd data/
$ wget -c https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz && tar -xzvf is2res_train_val_test_lmdbs.tar.gz
```

Process the data (i.e., use the 10k version).
``` bash
$ mv is2res_train_val_test_lmdbs/data/is2re/all/train is2res_train_val_test_lmdbs/data/is2re/all/train_all
$ mv is2res_train_val_test_lmdbs/data/is2re/10k/train is2res_train_val_test_lmdbs/data/is2re/all/train
```

Run hyperparameter tuning
``` bash
$ source run_tuning.sh
```

Process the results using `run_tuning.ipynb`. 
