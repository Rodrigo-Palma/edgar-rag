# make demo, replayed from eval/ci/tape with no model.
# Re-record from the repo root after `git lfs pull` and `uv sync --frozen`:
#   make demo-record
# Re-record in the same commit as any change to scripts/demo.sh, eval/ci/tape,
# eval/ci/index or the response schema, so the GIF never shows an older service.
Output docs/media/demo.gif

Require make
Require uv

Set Shell "bash"
Set FontSize 15
Set Width 1160
Set Height 440
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Framerate 12

Hide
Type "export EDGAR_RAG_DEMO_VIEW=brief PS1='$ '" Enter
Type "clear" Enter
Show

Type "make demo" Sleep 500ms Enter
Wait+Screen@90s /demo: one answer/
Sleep 8s
