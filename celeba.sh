#!/bin/bash -l
#SBATCH --job-name=quantised_sa_od_celeba_end_to_end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=1
##SBATCH --time=0-0:05:00
#SBATCH --partition=GTX780
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
                     ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/quantised_sa;
      python3 -u quantised_sa_rep/training_od.py --dataset 'celeba' --task 'celeba_end_to_end' --device 'gpu' --max_epochs 1000 --batch_size 64 --train_path "/home/quantised_sa/datasets/celeba/celeba/CelebA" --seed 0 --nums 8 8 8 8 --num_workers 4;
      free -m;
";

singularity instance stop ml_env