#!/bin/bash -l
#SBATCH --job-name=sa_od_clevr_seed11_end_to_end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=DGX-1v100
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --no-kill
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/quantised_sa:/home/quantised_sa \
                     ml_env.sif ml_env5

singularity exec instance://ml_env5 /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/quantised_sa;
      python3 -u quantised_sa_rep/training_od.py --dataset 'clevr' --no-quantization --task 'clevr slot attention 22:38 2.02.2023' --beta 1 --device 'gpu' --max_epochs 2000 --batch_size 64 --train_path "/home/quantised_sa/datasets/clevr" --val_path "/home/quantised_sa/datasets/clevr_with_masks/clevr_with_masks" --seed 11 --nums 8 3 2 2 --num_workers 4;
      free -m;
";

singularity instance stop ml_env5