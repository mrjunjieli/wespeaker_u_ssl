#!/bin/bash
set -euo pipefail

exp_dir=
model_path=
nj=1
gpus="[2]"
data_type=shard
data=data

. tools/parse_options.sh

for dset in vox2_dev vox1; do
  wavs_num=$(wc -l < "${data}/${dset}/wav.scp")
  bash tools/extract_embedding_uncertainty.sh \
    --exp_dir "${exp_dir}" \
    --model_path "${model_path}" \
    --data_type "${data_type}" \
    --data_list "${data}/${dset}/${data_type}.list" \
    --wavs_num "${wavs_num}" \
    --store_dir "${dset}" \
    --batch_size 1 \
    --num_workers 1 \
    --nj "${nj}" \
    --gpus "${gpus}"
done
