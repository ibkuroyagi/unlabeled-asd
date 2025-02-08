# Unlabeled condition anomalous sound detection

## Environmental setting

```bash
cd tools
make
```

## Dataset

Download all the data from the dcase2023 task2 dataset and arrange it as follows.

```bash
dev_data
├── bandsaw
│   ├── attributes_00.csv
│   ├── test
│   │   ├── section_00_source_test_anomaly_0005_vel_15.wav
│   │   ├── section_00_source_test_anomaly_0016_vel_15.wav
│   └── train
├── bearing
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── fan
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── gearbox
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── grinder
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── shaker
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── slider
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── ToyCar
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── ToyDrone
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── ToyNscale
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── ToyTank
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── ToyTrain
│   ├── attributes_00.csv
│   ├── test
│   └── train
├── Vacuum
│   ├── attributes_00.csv
│   ├── test
│   └── train
└── valve
    ├── attributes_00.csv
    ├── test
    └── train

```

## Command

```bash
# Set your path to audioset dir in run.sh
./run.sh
# By specifying a stage, you can execute a job from the middle of the stage.
./run.sh --stage 2 --stop_stage 5
```

## Citation
