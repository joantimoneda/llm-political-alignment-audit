# Persona scripts

One file per topic. Sections are in `[brackets]`, each non-blank line is one
user turn, and lines starting with `#` are comments. `01_collect.py` checks
every file against `config.TOPICS` before any API call: the cue sections must
match `cue_code` exactly, and a missing section is an error.

Sections:

- `[cue <level>]`: one block per cue level. `[cue cold]` exists and is empty;
  it is the no-persona baseline. Claiming to be undecided is itself a cue,
  which is what the `neutral` level is for.
- `[probe]`: exactly one line, the direct question.
- `[pressure]`: the escalation ladder, one step per line.
- `[profiling]`: the "where do I fall on a 0 to 10 scale" turns. Absent for
  the control topic.
- `[spillover <topic>]`: probe and pressure for a second topic the user never
  raised (abortion to guns), scored on its own scale.

All scripts follow the seven-turn arc of the abortion script, which is the
pilot script verbatim: greeting, frame, three substantive turns, a summary
position ("Thanks for all your thoughts. ..."), and a close. Matching the arc
keeps cue length constant across topics, so differences in mirroring are not
differences in how much the user said.

The cue axis is topic-specific (ideology, view of the evidence, sympathy,
union or independence). Its direction is coded in `config.TOPICS[...]["cue_code"]`
with positive values on the pole that scores high on the topic's stance scale,
so a positive mirroring slope always means the model moved toward the user.
Nazism has two cue levels (neutral, sympathetic), so its mirroring is a
contrast rather than a slope. The vaccines script was collected after the main
run as an appendix.

A hash of the persona file is recorded with every collected cell, so the
files must not be edited after collection.
