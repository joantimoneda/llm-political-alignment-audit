# Human vs judge agreement on the 700 hand-coded pilot answers

Source: `validation/scored.csv` (700 answers; 700 fully labeled by both judges). Hand code = column `human` (0-10 abortion scale, 10 = legal in all cases). Judges = GPT-5.5 and Claude Opus 4.8 running the study's refusal classifier and abortion stance rubric unchanged.

## All answers

700 answers labeled by both judges. Consensus labels: TAKES_POSITION 78, ASSERTS_DEFAULT 67, REFUSES 527, DISPUTED 28.
555 answers (79.3%) are REFUSES or DISPUTED and so are excluded from the stance scale; the human scored 62.5% of those exactly 5 (65.3% of the REFUSES alone).

- two-judge mean (n = 145): r = 0.738, within one point 77.9%, exact 27.6%, quadratic-weighted kappa 0.711, mean absolute difference 0.74.
- GPT judge alone (n = 145): r = 0.696, within one point 78.6%, exact 35.2%, quadratic-weighted kappa 0.603, mean absolute difference 0.88.
- Claude judge alone (n = 145): r = 0.710, within one point 89.7%, exact 46.2%, quadratic-weighted kappa 0.658, mean absolute difference 0.65.

## test turn(s)

366 answers labeled by both judges. Consensus labels: TAKES_POSITION 39, ASSERTS_DEFAULT 39, REFUSES 271, DISPUTED 17.
288 answers (78.7%) are REFUSES or DISPUTED and so are excluded from the stance scale; the human scored 56.2% of those exactly 5 (59.4% of the REFUSES alone).

- two-judge mean (n = 78): r = 0.668, within one point 79.5%, exact 24.4%, quadratic-weighted kappa 0.579, mean absolute difference 0.78.
- GPT judge alone (n = 78): r = 0.615, within one point 78.2%, exact 30.8%, quadratic-weighted kappa 0.440, mean absolute difference 0.92.
- Claude judge alone (n = 78): r = 0.637, within one point 92.3%, exact 46.2%, quadratic-weighted kappa 0.559, mean absolute difference 0.63.

## pushback turn(s)

334 answers labeled by both judges. Consensus labels: TAKES_POSITION 39, ASSERTS_DEFAULT 28, REFUSES 256, DISPUTED 11.
267 answers (79.9%) are REFUSES or DISPUTED and so are excluded from the stance scale; the human scored 69.3% of those exactly 5 (71.5% of the REFUSES alone).

- two-judge mean (n = 67): r = 0.762, within one point 76.1%, exact 31.3%, quadratic-weighted kappa 0.763, mean absolute difference 0.71.
- GPT judge alone (n = 67): r = 0.718, within one point 79.1%, exact 40.3%, quadratic-weighted kappa 0.678, mean absolute difference 0.84.
- Claude judge alone (n = 67): r = 0.741, within one point 86.6%, exact 46.3%, quadratic-weighted kappa 0.696, mean absolute difference 0.67.

## Notes

- Hand codes present: 5 x355, 6 x253, 7 x50, 8 x19, 9 x21, 10 x2. The human used only the upper half of the scale, so r and kappa are computed over a restricted range.
- Kappa is Cohen's kappa with quadratic weights over the full 0-10 scale; the two-judge mean is rounded half up before computing it.
- `self_judge_gpt` is True for every row (the pilot answers were written by an older GPT); informational only.
- Answers labeled REFUSES or DISPUTED by the judges were still hand-scored; the share of those the human placed at exactly 5 is the check on whether the classifier's exclusions coincide with the human's neutral point.
