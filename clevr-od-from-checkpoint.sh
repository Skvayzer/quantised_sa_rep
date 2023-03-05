#!/bin/bash -l
#SBATCH --job-name=quantised_sa_od_clevr_end_to_end_from_sa
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
                     ml_env.sif ml_env7

singularity exec instance://ml_env7 /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/quantised_sa;
      python3 -u quantised_sa_rep/training_od.py --dataset 'clevr' --from_checkpoint '/home/quantised_sa/sa_autoencoder_end_to_end/clevr/clevr from SA 24.02.2023/545_0.015625_clevr from SA 24.02.2023_clevr_od_pretrained.ckpt' --task 'clevr VQ-SA from SA 05.03.2023 (ckpt)' --beta 0.015625 --device 'gpu' --max_epochs 2000 --batch_size 64 --train_path "/home/quantised_sa/datasets/sa_clevr/clevr_with_masks_6" --seed 31 --nums 8 3 2 2 --num_workers 4;
      free -m;
";

singularity instance stop ml_env7