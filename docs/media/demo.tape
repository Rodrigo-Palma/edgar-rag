# A tour of edgar-rag in six scenes, every line the real output of the
# command typed above it, replayed from eval/ci/tape with no model.
# Re-record from the repo root after `git lfs pull` and `uv sync --frozen`:
#   make demo-record
# Re-record in the same commit as any change to scripts/demo_tour.sh,
# scripts/demo_brief.py, scripts/print_result.py, eval/ci/tape, eval/ci/index,
# docs/eval/report-v1.md or the response schema, so the GIF never shows an
# older service. `ask` is defined in scripts/demo_tour.sh.
Output docs/media/demo.gif

Require make
Require uv
Require git
Require curl

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 640
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms
Set Framerate 10
Set CursorBlink false
Set WaitTimeout 90s
Set WaitPattern /\$$/

Hide
Type "export PS1='$ ' && source scripts/demo_tour.sh && clear" Enter
Wait
Show

Type "# 1. An answer that cites the filing it came from" Enter
Type "ask 2025 'How much revenue did Apple report for fiscal year 2025?'" Sleep 300ms Enter
Wait
Sleep 6s

Hide
Type "clear" Enter
Show
Type "# 2. A question about another year: declined before the model is called" Enter
Type "ask 2025 'What cash dividends did Apple pay to shareholders in fiscal 2020?'" Sleep 300ms Enter
Wait
Sleep 6s

Hide
Type "clear" Enter
Show
Type "# 3. A question the filing cannot answer: the model declines" Enter
Type `ask 2025 "What was Apple's investment banking revenue in fiscal 2024?"` Sleep 300ms Enter
Wait
Sleep 6s

Hide
Type "clear" Enter
Show
Type "# 4. The evaluation report rebuilds from the frozen run" Enter
Type "make eval" Sleep 300ms Enter
Wait
Type "git status --porcelain docs/eval" Sleep 300ms Enter
Wait
Type "git diff --exit-code docs/eval && echo 'report reproduces byte for byte'" Sleep 300ms Enter
Wait
Sleep 5s

Hide
Type "clear" Enter
Show
Type "# 5. The result, printed from that report" Enter
Type "make result" Sleep 300ms Enter
Wait
Sleep 14s

Hide
Type "clear" Enter
Show
Type "# 6. The CI gate: the dev split replayed against its baseline" Enter
Type "make eval-ci 2>&1 | tail -1" Sleep 300ms Enter
Wait
Sleep 5s

Hide
Type "tour_stop" Enter
