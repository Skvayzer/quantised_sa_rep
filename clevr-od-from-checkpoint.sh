#!/bin/bash -l
#SBATCH --job-name=quantised_sa_od_clevr_ari_calc_end_to_end_checkpoint1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=titan_X
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
      python3 -u quantised_sa_rep/training_od.py --dataset 'clevr' --task 'test 30.01' --from_checkpoint '/home/quantised_sa/sa_autoencoder_end_to_end/clevr/clevr vanilla 21:15 28.01.2023/80_0.0_clevr vanilla 21:15 28.01.2023_clevr_od_pretrained.ckpt' --beta 0 --device 'gpu' --max_epochs 2000 --batch_size 16 --train_path "/home/quantised_sa/datasets/clevr" --seed 37 --nums 8 3 2 2 --num_workers 4;
      free -m;
";

singularity instance stop ml_env7