#!/usr/bin/env bash
# Follow-up queue, 2026-10-19 (after the holiday queue; see RESEARCH/paper-log.md, 2026-10-19).
#   1. scale confound: full-face alone at higher scales, ours alone at higher scales
#   2. fusion with the trained-longer adapter (105k cool-down) + grid extension past (0.8, 0.5)
#   3. fru2fru for the leading fusions
#   4. Stage A continued: 100k → 150k, cool-down 150k → 155k, then evaluated
#
#   cd ~/WPU/testings/faceid-adapter && nohup bash queue_followup.sh > queue2.log 2>&1 & disown
# Same mechanics as queue_holiday.sh: logs/<step>.log, .queue/<step>.done, rerun skips done steps.
set -u
cd "$(dirname "$0")"
mkdir -p logs .queue

P=configs/gpu.yaml
PE=configs/gpu_plus.yaml
EVALSET=~/WPU/testings/3-faces/evalset.json
W=faceid_aura_sd15.safetensors
A50=checkpoints/poc/step-0050000/$W
A105=checkpoints/real_v3_cd100/step-0105000/$W
EV="python eval2.py --config configs/poc.yaml --paths-from $PE --evalset $EVALSET --split dev"
OUTA=checkpoints/ev2/dev_reg2fru
OUTB=checkpoints/ev2/dev_fru2fru

step() {
  local name=$1; shift
  if [ -f ".queue/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" > "logs/$name.log" 2>&1; then
    touch ".queue/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (exit $?, see logs/$name.log)"
  fi
}

# 1. is the fusion gain more than "more total IP strength"?
step f_full_scales $EV --protocol reg2fru --out $OUTA --modes full-face@0.5 full-face@1.0 full-face@1.2 --baseline full-face
step f_105k_A $EV --protocol reg2fru --out $OUTA --adapter $A105 --tag 105k --baseline full-face \
  --modes ours@1.0 ours@1.2 ours+full-face@0.6,0.5 ours+full-face@0.8,0.5 ours+full-face@1.0,0.5 \
          ours+full-face@0.8,0.7 ours+full-face@1.0,0.7
step f_50k_ext_A $EV --protocol reg2fru --out $OUTA --adapter $A50 --tag 50k --baseline full-face \
  --modes ours+full-face@1.0,0.5 ours+full-face@0.8,0.7 ours+full-face@1.0,0.7

# 1b. framing confound: tighter h94 crop (1.4x box = face only), so full-face draws a close-up too
step f_crop_A $EV --protocol reg2fru --out $OUTA --adapter $A105 --tag 105k --baseline full-face --crop-scale 1.4   --modes full-face@0.8 full-face@0.5 ours+full-face@0.8,0.5 ours+full-face@0.6,0.3

# 2. second protocol for the leaders
step f_fullB $EV --protocol fru2fru --out $OUTB --modes full-face@0.5 --baseline full-face
step f_105k_B $EV --protocol fru2fru --out $OUTB --adapter $A105 --tag 105k --baseline full-face \
  --modes ours ours+full-face@0.8,0.5 ours+full-face@1.0,0.7
step f_50k_B $EV --protocol fru2fru --out $OUTB --adapter $A50 --tag 50k --baseline full-face \
  --modes ours+full-face@0.8,0.5

# 3. Stage A continued (~9 h) + cool-down (~1 h); resumes from the newest state in checkpoints/real_v3
step train_150k python train.py --config configs/real_v3.yaml --paths-from $P --resume latest --max-steps 150000
step train_cd150 python train.py --config configs/real_v3.yaml --paths-from $P --resume checkpoints/real_v3/state-0150000 \
  --out checkpoints/real_v3_cd150 --decay-start 150000 --max-steps 155000
for c in "150k checkpoints/real_v3/step-0150000" "155k checkpoints/real_v3_cd150/step-0155000"; do
  set -- $c
  step "ev_$1_A" $EV --protocol reg2fru --out $OUTA --adapter $2/$W --tag $1 --baseline full-face \
    --modes ours ours+full-face@0.8,0.5
  step "ev_$1_B" $EV --protocol fru2fru --out $OUTB --adapter $2/$W --tag $1 --baseline full-face --modes ours
done

# 4. final summaries
for o in $OUTA $OUTB; do
  for b in full-face "ours[50k]"; do
    python eval2.py --evalset $EVALSET --out $o --modes ours --baseline "$b" --summarize-only > /dev/null 2>&1 \
      && cp $o/summary.json "$o/summary_vs_$(echo $b | tr -d '[]')".json
  done
done
echo "$(date '+%F %T')  queue finished"
