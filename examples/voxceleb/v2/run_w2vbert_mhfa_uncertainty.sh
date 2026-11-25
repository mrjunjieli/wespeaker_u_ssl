#!/bin/bash
set -euo pipefail

stage=3
stop_stage=6
data=data
data_type=shard
config_s1=conf/w2vbert-mhfa-uncertainty-s1_frozen.yaml
config_s2=conf/w2vbert-mhfa-uncertainty-s2_ft.yaml
config_s3=conf/w2vbert-mhfa-uncertainty-s3_lmft.yaml
exp_dir_s1=exp/w2vbert-mhfa-uncertainty-s1_frozen
exp_dir_s2=exp/w2vbert-mhfa-uncertainty-s2_ft
exp_dir_s3=exp/w2vbert-mhfa-uncertainty-s3_lmft
gpus="[2]"
nj=1
checkpoint=
num_avg_s1=4
num_avg_s2=1
num_avg_s3=1
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
help_message="Usage: bash run_w2vbert_mhfa_uncertainty.sh --stage N --stop_stage N\n1: prepare data; 2: build lists/LMDB\n3-6: s1_frozen train/average/extract/cosine\n7-10: s2_ft train/average/extract/cosine\n11-14: s3_lmft (gap-margin) train/average/extract/cosine\nUse --checkpoint /path/model_N.pt to resume the selected training stage."

cd "$(dirname "${BASH_SOURCE[0]}")"
. ./path.sh
. tools/parse_options.sh || exit 1
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

if [[ ! ${stage} =~ ^[0-9]+$ || ! ${stop_stage} =~ ^[0-9]+$ ]] ||
   [ "${stage}" -lt 1 ] || [ "${stop_stage}" -gt 14 ] ||
   [ "${stage}" -gt "${stop_stage}" ]; then
  echo "Expected 1 <= stage <= stop_stage <= 14" >&2
  exit 1
fi
if [[ ${gpus} != "[2]" && ${gpus} != "[2,3]" ]]; then
  echo "Only physical GPU 2 or GPUs 2,3 are permitted." >&2
  exit 1
fi
if [[ ${data_type} != shard && ${data_type} != raw ]]; then
  echo "data_type must be shard or raw" >&2
  exit 1
fi
if [ "${stage}" -le 2 ] && [ -L "${data}" ]; then
  echo "Refusing to regenerate shared data through a symlink. Use --data with a new directory, or start at stage 3." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$(echo "${gpus}" | tr -d '[]')"
num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
train_gpus="[0]"
[ "${num_gpus}" -eq 1 ] || train_gpus="[0,1]"
if [ -n "${checkpoint}" ]; then
  if [ "${stage}" -ne 3 ] && [ "${stage}" -ne 7 ] && [ "${stage}" -ne 11 ]; then
    echo "Resume requires --stage 3, 7 or 11." >&2
    exit 1
  fi
  if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 7 ] ||
     [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 11 ]; then
    echo "Resume only one training phase per invocation." >&2
    exit 1
  fi
fi
if { [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; } ||
   { [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; } ||
   { [ "${stage}" -le 11 ] && [ "${stop_stage}" -ge 11 ]; }; then
  for gpu in $(echo "${gpus}" | tr -d '[]' | tr ',' ' '); do
    nvidia-smi -i "${gpu}" --query-gpu=index,memory.used,memory.total --format=csv
  done
  python -c "import torch; assert torch.cuda.device_count() >= ${num_gpus}, 'Requested physical GPU is unavailable; use --gpus [2] if GPU 3 is absent'"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  bash local/prepare_data.sh --stage 2 --stop_stage 4 --data "${data}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  for dset in vox2_dev vox1; do
    if [ "${data_type}" == shard ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 --num_threads 16 \
        "${data}/${dset}/wav.scp" "${data}/${dset}/utt2spk" \
        "${data}/${dset}/shards" "${data}/${dset}/shard.list"
    else
      python tools/make_raw_list.py "${data}/${dset}/wav.scp" \
        "${data}/${dset}/utt2spk" "${data}/${dset}/raw.list"
    fi
  done
  python tools/make_lmdb.py "${data}/musan/wav.scp" "${data}/musan/lmdb"
  python tools/make_lmdb.py "${data}/rirs/wav.scp" "${data}/rirs/lmdb"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo "Stage 3: frozen W2V-BERT, train NoAdapter diagonal Gaussian backend"
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    wespeaker/bin/train.py --config "${config_s1}" \
    --exp_dir "${exp_dir_s1}" --gpus "${train_gpus}" \
    --num_avg "${num_avg_s1}" --data_type "${data_type}" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
  python wespeaker/bin/average_model.py --dst_model "${exp_dir_s1}/models/avg_model.pt" \
    --src_path "${exp_dir_s1}/models" --num "${num_avg_s1}"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  bash local/extract_vox_w2vbert_mhfa_uncertainty.sh --exp_dir "${exp_dir_s1}" \
    --model_path "${exp_dir_s1}/models/avg_model.pt" \
    --data "${data}" --data_type "${data_type}" --nj "${nj}" --gpus "${gpus}"
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
  bash -e -o pipefail local/score.sh --stage 1 --stop_stage 2 \
    --data "${data}" --exp_dir "${exp_dir_s1}" --trials "${trials}"
fi

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
  echo "Stage 7: jointly fine-tune W2V-BERT and Gaussian backend"
  if [ -z "${checkpoint}" ]; then
    test -s "${exp_dir_s1}/models/avg_model.pt"
    test ! -e "${exp_dir_s2}/models"
    mkdir -p "${exp_dir_s2}/models"
    cp "${exp_dir_s1}/models/avg_model.pt" "${exp_dir_s2}/models/model_0.pt"
  fi
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    wespeaker/bin/train.py --config "${config_s2}" \
    --exp_dir "${exp_dir_s2}" --gpus "${train_gpus}" \
    --num_avg "${num_avg_s2}" --data_type "${data_type}" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    --checkpoint "${checkpoint:-${exp_dir_s2}/models/model_0.pt}"
fi

if [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
  python wespeaker/bin/average_model.py --dst_model "${exp_dir_s2}/models/avg_model.pt" \
    --src_path "${exp_dir_s2}/models" --num "${num_avg_s2}"
fi

if [ "${stage}" -le 9 ] && [ "${stop_stage}" -ge 9 ]; then
  bash local/extract_vox_w2vbert_mhfa_uncertainty.sh --exp_dir "${exp_dir_s2}" \
    --model_path "${exp_dir_s2}/models/avg_model.pt" \
    --data "${data}" --data_type "${data_type}" --nj "${nj}" --gpus "${gpus}"
fi

if [ "${stage}" -le 10 ] && [ "${stop_stage}" -ge 10 ]; then
  bash -e -o pipefail local/score.sh --stage 1 --stop_stage 2 \
    --data "${data}" --exp_dir "${exp_dir_s2}" --trials "${trials}"
fi

if [ "${stage}" -le 11 ] && [ "${stop_stage}" -ge 11 ]; then
  echo "Stage 11: long chunks, gap-adaptive margin with detached uncertainty scaling (see ${config_s3})"
  if [ -z "${checkpoint}" ]; then
    test -s "${exp_dir_s2}/models/avg_model.pt"
    test ! -e "${exp_dir_s3}/models"
    mkdir -p "${exp_dir_s3}/models"
    cp "${exp_dir_s2}/models/avg_model.pt" "${exp_dir_s3}/models/model_0.pt"
  fi
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    wespeaker/bin/train.py --config "${config_s3}" \
    --exp_dir "${exp_dir_s3}" --gpus "${train_gpus}" \
    --num_avg "${num_avg_s3}" --data_type "${data_type}" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    --checkpoint "${checkpoint:-${exp_dir_s3}/models/model_0.pt}"
fi

if [ "${stage}" -le 12 ] && [ "${stop_stage}" -ge 12 ]; then
  python wespeaker/bin/average_model.py --dst_model "${exp_dir_s3}/models/avg_model.pt" \
    --src_path "${exp_dir_s3}/models" --num "${num_avg_s3}"
fi

if [ "${stage}" -le 13 ] && [ "${stop_stage}" -ge 13 ]; then
  bash local/extract_vox_w2vbert_mhfa_uncertainty.sh --exp_dir "${exp_dir_s3}" \
    --model_path "${exp_dir_s3}/models/avg_model.pt" \
    --data "${data}" --data_type "${data_type}" --nj "${nj}" --gpus "${gpus}"
fi

if [ "${stage}" -le 14 ] && [ "${stop_stage}" -ge 14 ]; then
  bash -e -o pipefail local/score.sh --stage 1 --stop_stage 2 \
    --data "${data}" --exp_dir "${exp_dir_s3}" --trials "${trials}"
fi
