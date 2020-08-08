## preparing dataset into tfrecord format-
https://github.com/google/youtube-8m/tree/master/feature_extractor refer to this for converting mp4 video files into tfrecord format suitable for training 

## Training the single models

Each of the following command lines train a single model. They are scheduled to stop training at the good time.


Each model takes several days to train, so each command line are separated in order to be run in parallel if possible. 
Please replace 'path_to_features' with the folder path which contains all the tensorflow record frame level feature.
```sh
path_to_features='path_to_features'
```

Training Gated NetVLAD (256 Clusters):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=NetVLADModelLF --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --max_step=700000
```

Note: The best single model is this one but with the flag --max_step=300000. We somehow need it to train longer for better effect on the ensemble.
G
Training Gated NetFV (128 Clusters):


```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=NetFVModelLF --train_dir=gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --fv_cluster_size=128 --fv_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --fv_relu=False --gating=True --moe_prob_gating=True --fv_couple_weights=False --max_step=600000
```

Training Gated Soft-DBoW (4096 Clusters):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=GatedDbofModelLF --train_dir=gateddboflf-4096-1024-80-0002-300iter --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --dbof_cluster_size=4096 --dbof_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --dbof_relu=False --max_step=1000000
```

Training Soft-DBoW (8000 Clusters):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=SoftDbofModelLF --train_dir=softdboflf-8000-1024-80-0002-300iter --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --dbof_cluster_size=8000 --dbof_hidden_size=1024 --iterations=300 --dbof_relu=False --max_step=800000
```

Training Gated NetRVLAD (256 Clusters):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=NetVLADModelLF --train_dir=gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --lightvlad=True --max_step=600000
```

Training GRU (2 layers):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=GruModel --train_dir=GRU-0002-1200 --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=128 --base_learning_rate=0.0002 --gru_cells=1200 --learning_rate_decay=0.9 --moe_l2=1e-6 --max_step=300000
```

Training LSTM (2 layers):

```sh
python train.py --train_data_pattern="$path_to_features/*a*??.tfrecord" --model=LstmModel --train_dir=lstm-0002-val-150-random --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=128 --base_learning_rate=0.0002 --iterations=150 --lstm_random_sequence=True --max_step=400000
```


## Inference

After training, we will write the predictions into 7 different files and then ensemble them.
Run each one of this command to run inference for each model.

```sh
python inference.py --output_file=test-lstm-0002-val-150-random.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=LstmModel --train_dir=lstm-0002-val-150-random --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --iterations=150 --lstm_random_sequence=True --run_once=True --top_k=50

python inference.py --output_file=test-gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=NetVLADModelLF --train_dir=gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --lightvlad=True --run_once=True  --top_k=50 

python inference.py --output_file=test-gateddboflf-4096-1024-80-0002-300iter-gatedmoe.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=GatedDbofModelLF --train_dir=gateddboflf-4096-1024-80-0002-300iter-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=512 --base_learning_rate=0.0002 --dbof_cluster_size=4096 --dbof_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --dbof_relu=False --moe_prob_gating=True --run_once=True --top_k=50

python inference.py --output_file=test-gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=NetFVModelLF --train_dir=gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --fv_cluster_size=128 --fv_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --fv_relu=False --gating=True --moe_prob_gating=True --fv_couple_weights=False --top_k=50

python inference.py --output_file=test-GRU-0002-1200.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=GruModel --train_dir=GRU-0002-1200 --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --gru_cells=1200 --learning_rate_decay=0.9 --moe_l2=1e-6 --run_once=True --top_k=50

python inference.py --output_file=test-gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=NetVLADModelLF --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True  --top_k=50

python inference.py --output_file=test-softdboflf-8000-1024-80-0002-300iter.csv --input_data_pattern="$path_to_features/test*.tfrecord"  --model=SoftDbofModelLF --train_dir=softdboflf-8000-1024-80-0002-300iter --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=256 --base_learning_rate=0.0002 --dbof_cluster_size=8000 --dbof_hidden_size=1024 --iterations=300 --dbof_relu=False --run_once=True --top_k=50
```

## Averaging the models

After inference done for all models just run:


```sh
python file_averaging.py
```


