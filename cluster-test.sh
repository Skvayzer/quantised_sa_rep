#!/bin/bash -l
#SBATCH --job-name=quantised_sa_sp_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=DGX-1v100
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=korchemnyi.av@phystech.edu
#SBATCH --mail-type=END
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/quantised_sa:/home/quantised_sa \
                     --env WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd \
                     ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      wandb login --relogin c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      nvidia-smi;
      free -m;
      cd /home/quantised_sa;
      python3 quantised_sa_rep/training_od.py --dataset 'clevr-tex' --device 'gpu' --max_epochs 442 --batch_size 512 --train_path "/home/quantised_sa/datasets/clevr-tex" --seed 0 --nums 8 8 8 8 --num_workers 4 ;
      free -m;
";

singularity instance stop ml_env