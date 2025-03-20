#!/bin/bash

# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
stage=0
stop_stage=8
seed=0
epoch=50
outdir=exp
audioset_dir="<path to your audioset dir>"
. ./utils/parse_options.sh

n_max=0
use_att=wodo
d=1
d1_tag=${outdir}/d${d}_${use_att}_max${n_max}/seed${seed}
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Start 0: Training baseline model."
    log "${d1_tag}/train.log"
    ${cuda_cmd} --gpu "1" "${d1_tag}/train.log" \
        python train_infer.py \
        --tag "${d1_tag}" \
        --use_att "${use_att}" \
        --seed "${seed}" \
        --dumy_dir "" \
        --audio_dir "" \
        --n_max "${n_max}"
fi
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Start 1: Inference audioset data."
    model_dir=${d1_tag}/epoch${epoch}
    for segments in balanced_train_segments eval_segments unbalanced_train_segments; do
        log ${model_dir}/${segments}.log
        ${cuda_cmd} --gpu "1" "${model_dir}/${segments}.log" \
            python core_set_selection.py \
            --audioset_dir ${audioset_dir} \
            --model_dir ${model_dir} \
            --segments ${segments} \
            --use_att ${use_att} \
            --seed ${seed}
    done
fi
use_att=dumy
d=2
d2_n_max=1000
d2_tag=${outdir}/d${d}_${use_att}_max${d2_n_max}/seed${seed}
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Start 2: Re-training model with pseudo-label and core-set external data."
    log "${d2_tag}/train.log"
    prev_dir="${d1_tag}/epoch${epoch}"
    dumy_dir="${prev_dir}"
    audio_dir="${prev_dir}"
    ${cuda_cmd} --gpu "1" "${d2_tag}/train.log" \
        python train_infer.py \
        --tag "${d2_tag}" \
        --use_att "${use_att}" \
        --seed "${seed}" \
        --dumy_dir "${dumy_dir}" \
        --audio_dir "${audio_dir}" \
        --n_max "${d2_n_max}"
fi
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Start 3: Inference audioset data."
    model_dir=${d2_tag}/epoch${epoch}
    for segments in balanced_train_segments eval_segments unbalanced_train_segments; do
        log ${model_dir}/${segments}.log
        ${cuda_cmd} --gpu "1" "${model_dir}/${segments}.log" \
            python core_set_selection.py \
            --audioset_dir ${audioset_dir} \
            --model_dir ${model_dir} \
            --segments ${segments} \
            --use_att ${use_att} \
            --seed ${seed}
    done
fi
d=3
d3_n_max=1000
d3_tag=${outdir}/d${d}_${use_att}_max${d3_n_max}/seed${seed}
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Start 4: Re-training model with pseudo-label and core-set external data."
    log "${d3_tag}/train.log"
    prev_dir="${d2_tag}/epoch${epoch}"
    dumy_dir="${prev_dir}"
    audio_dir="${prev_dir}"
    ${cuda_cmd} --gpu "1" "${d3_tag}/train.log" \
        python train_infer.py \
        --tag "${d3_tag}" \
        --use_att "${use_att}" \
        --seed "${seed}" \
        --dumy_dir "${dumy_dir}" \
        --audio_dir "${audio_dir}" \
        --n_max "${d3_n_max}"
fi
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Start 5: Re-inference audioset data."
    model_dir=${d3_tag}/epoch${epoch}
    for segments in balanced_train_segments eval_segments unbalanced_train_segments; do
        log ${model_dir}/${segments}.log
        ${cuda_cmd} --gpu "1" "${model_dir}/${segments}.log" \
            python core_set_selection.py \
            --audioset_dir ${audioset_dir} \
            --model_dir ${model_dir} \
            --segments ${segments} \
            --use_att ${use_att} \
            --seed ${seed}
    done
fi
d=4
d4_n_max=1000
d4_tag=${outdir}/d${d}_${use_att}_max${d4_n_max}/seed${seed}
if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    log "Start 6: Re-training model with pseudo-label and core-set external data."
    prev_dir="${d3_tag}/epoch${epoch}"
    dumy_dir="${prev_dir}"
    audio_dir="${prev_dir}"
    log "${d4_tag}/train.log"
    ${cuda_cmd} --gpu "1" "${d4_tag}/train.log" \
        python train_infer.py \
        --tag "${d4_tag}" \
        --use_att "${use_att}" \
        --seed "${seed}" \
        --dumy_dir "${dumy_dir}" \
        --audio_dir "${audio_dir}" \
        --n_max "${d4_n_max}"
fi
if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    log "Start 7: Re-inference audioset data."
    model_dir=${d4_tag}/epoch${epoch}
    for segments in balanced_train_segments eval_segments unbalanced_train_segments; do
        log ${model_dir}/${segments}.log
        ${cuda_cmd} --gpu "1" "${model_dir}/${segments}.log" \
            python core_set_selection.py \
            --audioset_dir ${audioset_dir} \
            --model_dir ${model_dir} \
            --segments ${segments} \
            --use_att ${use_att} \
            --seed ${seed}
    done
fi
d=5
d5_n_max=1000
d5_tag=${outdir}/d${d}_${use_att}_max${d5_n_max}/seed${seed}
if [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
    log "Start 8: Re-training model with pseudo-label and core-set external data."
    prev_dir="${d4_tag}/epoch${epoch}"
    dumy_dir="${prev_dir}"
    audio_dir="${prev_dir}"
    log "${d4_tag}/train.log"
    ${cuda_cmd} --gpu "1" "${d4_tag}/train.log" \
        python train_infer.py \
        --tag "${d5_tag}" \
        --use_att "${use_att}" \
        --seed "${seed}" \
        --dumy_dir "${dumy_dir}" \
        --audio_dir "${audio_dir}" \
        --n_max "${d5_n_max}"
fi
log "Finish all stage!!"
