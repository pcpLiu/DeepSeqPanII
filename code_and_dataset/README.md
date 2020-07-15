# Tested Environment
All codes are tested under below environment
- Python >= 3.5
- PyTorch == 0.4.1
- Linux & Mac OS

# Training model on BD2013
```
$ python run.py "BD2013/config.json"
```

# Training LOMO models on BD2016
```
$ ./run_LOMO_2016.sh
```

# Quick test a sample
```
$ python main.py "../Models/benchmark_weekly/model_bd2013.pytorch" "DRA*01:01" "DRB1*01:01" "AAYSDQATPLLLSPR"
```
