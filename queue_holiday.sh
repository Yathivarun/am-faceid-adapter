#!/usr/bin/env bash
# Unattended queue for 2026-10-17 → holidays (plan 2026-10: Stage A, G1/G2 evals, Stage M).
#
#   source ~/WPU/testings/sketching/linux-wheelhouse/venv-master/bin/activate
#   cd ~/WPU/testings/faceid-adapter && nohup bash queue_holiday.sh > queue.log 2>&1 & disown
#
# Each step logs to logs/<step>.log and leaves .queue/<step>.done on success.
# A failed step is reported and the queue moves on. Rerunning the script
# skips finished steps, and training resumes from its latest state.
# Progress: tail -f queue.log ; ls .queue
set -u
cd "$(dirname "$0")"
mkdir -p logs .queue

P=configs/gpu.yaml                       # model paths (training)
PE=configs/gpu_plus.yaml                 # model paths + h94 full-face/plus-face (evaluation)
SET=~/WPU/testings/3-faces
EVALSET=$SET/evalset.json
A50=checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors
W=faceid_aura_sd15.safetensors
EV="python eval2.py --config configs/poc.yaml --paths-from $PE --evalset $EVALSET --split dev"
OUTA=checkpoints/ev2/dev_reg2fru         # every reg2fru result, compared per person
OUTB=checkpoints/ev2/dev_fru2fru

step() {                                 # step <name> <command...>
  local name=$1; shift
  if [ -f ".queue/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" > "logs/$name.log" 2>&1; then
    touch ".queue/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (exit $?, see logs/$name.log)"
  fi
}

train_main() {                           # Stage A: warm start 50k → 100k; resumes if interrupted
  if ls checkpoints/real_v3/state-* > /dev/null 2>&1; then
    python train.py --config configs/real_v3.yaml --paths-from $P --resume latest --keep-state-at 75000
  else
    python train.py --config configs/real_v3.yaml --paths-from $P --init-adapter $A50 --keep-state-at 75000
  fi
}

train_mix() {                            # Stage M: 50k → 65k with 20% SFHQ (control = real_v3 step 65000)
  if ls checkpoints/mix_sfhq20/state-* > /dev/null 2>&1; then
    python train.py --config configs/real_v3.yaml --paths-from $P --out checkpoints/mix_sfhq20 --max-steps 65000 \
      --extra-data data/faces_sfhq --extra-ratio 0.2 --extra-generic-captions --resume latest
  else
    python train.py --config configs/real_v3.yaml --paths-from $P --out checkpoints/mix_sfhq20 --max-steps 65000 \
      --extra-data data/faces_sfhq --extra-ratio 0.2 --extra-generic-captions --init-adapter $A50
  fi
}

# 0. evaluation set (skipped if it exists)
[ -f "$EVALSET" ] || step evalset python prep/build_evalset.py --root $SET/fru-images --dev $SET/dev60.txt \
  --aura "$(python -c "from train import load_config; print(load_config('configs/poc.yaml', {}, None, '$P')['paths']['aura'])")"   --out $EVALSET

# 1. baselines, eval v2 (G0): research 50k, full-face (production), plus-face
step ev_base_A $EV --protocol reg2fru --out $OUTA --adapter $A50 --tag 50k --modes ours full-face plus-face --baseline full-face
step ev_base_B $EV --protocol fru2fru --out $OUTB --adapter $A50 --tag 50k --modes ours full-face plus-face --baseline full-face

# 2. Stage A training (~9-10 h) and the two cool-down branches (~1 h each)
step train_main train_main
step train_cd75  python train.py --config configs/real_v3.yaml --paths-from $P --resume checkpoints/real_v3/state-0075000 \
  --out checkpoints/real_v3_cd75 --decay-start 75000 --max-steps 80000
step train_cd100 python train.py --config configs/real_v3.yaml --paths-from $P --resume checkpoints/real_v3/state-0100000 \
  --out checkpoints/real_v3_cd100 --decay-start 100000 --max-steps 105000

# 3. G1: the new checkpoints against the 50k adapter, same people and seeds
for c in "75k checkpoints/real_v3/step-0075000" "75k-cd checkpoints/real_v3_cd75/step-0080000" \
         "100k checkpoints/real_v3/step-0100000" "100k-cd checkpoints/real_v3_cd100/step-0105000"; do
  set -- $c
  step "ev_$1_A" $EV --protocol reg2fru --out $OUTA --adapter $2/$W --tag $1 --modes ours --baseline "ours[50k]"
  step "ev_$1_B" $EV --protocol fru2fru --out $OUTB --adapter $2/$W --tag $1 --modes ours --baseline "ours[50k]"
done

# 4. G2: fusion grid, research 50k + h94 at fixed scales (inference only)
GRID=""
for so in 0.4 0.6 0.8; do for sh in 0.2 0.3 0.5; do
  GRID="$GRID ours+full-face@$so,$sh ours+plus-face@$so,$sh"
done; done
step ev_fusion_A $EV --protocol reg2fru --out $OUTA --adapter $A50 --tag 50k --modes $GRID --baseline full-face

# 5. Stage M: 20% SFHQ mixed in, 15k steps from 50k; compared with real_v3 step 65000
step train_mix train_mix
step ev_65k_A  $EV --protocol reg2fru --out $OUTA --adapter checkpoints/real_v3/step-0065000/$W --tag 65k --modes ours --baseline "ours[50k]"
step ev_mix_A  $EV --protocol reg2fru --out $OUTA --adapter checkpoints/mix_sfhq20/step-0065000/$W --tag 65k-sfhq20 --modes ours --baseline "ours[65k]"

# 6. final summaries: every mode vs production full-face, and vs the 50k adapter
for o in $OUTA $OUTB; do
  for b in full-face "ours[50k]"; do
    python eval2.py --evalset $EVALSET --out $o --modes ours --baseline "$b" --summarize-only > /dev/null 2>&1       && cp $o/summary.json "$o/summary_vs_$(echo $b | tr -d '[]')".json
  done
done
echo "$(date '+%F %T')  queue finished"; ls .queue
