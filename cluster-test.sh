#!/bin/bash -l
#SBATCH --job-name=example
#SBATCH --nodes=1
#SBATCH --ntasks=16
##SBATCH --time=0-0:05:00
#SBATCH --partition=titan_X
##SBATCH --array=1-5
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/quantised_sa:/home/quantised_sa \
                     ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      set -x;
      nvidia-smi;
      free -m;
      cd //home/quantised_sa/;
      python3 quantized_sa_rep/training_od.py --dataset 'clevr-tex' --device 'gpu' --max_epochs 442 --batch_size 512 --train_path "/home/quantized_sa/datasets/clevr-tex" --seed 0 --nums 8 8 8 8 --num_workers 2 ;
      free -m;
" > output.txt

singularity instance stop ml_env