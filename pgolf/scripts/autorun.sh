#!/usr/bin/env bash
# Autonomous research runner via Claude Code CLI.
#
# Launches Claude Code with the /autorun skill to autonomously
# run experiments. Supports both general lmxlab research and
# Parameter Golf optimization.
#
# Usage:
#   ./scripts/autorun.sh [focus] [hours] [max-turns]
#
# Examples:
#   ./scripts/autorun.sh pgolf            # PGolf research for 1 hour
#   ./scripts/autorun.sh pgolf 2          # PGolf for 2 hours
#   ./scripts/autorun.sh pgolf 4 500      # PGolf for 4 hours, 500 turns
#   ./scripts/autorun.sh HYP-017 1        # Specific hypothesis for 1 hour
#   ./scripts/autorun.sh                  # General research for 1 hour
#
# Prerequisites:
#   - Claude Code CLI installed (claude command available)
#   - uv sync --extra hf --extra experiments
#   - For pgolf: parameter-golf repo cloned as sibling

set -euo pipefail

FOCUS="${1:-pgolf}"
HOURS="${2:-1}"
MAX_TURNS="${3:-200}"

# --- Calculate deadline timestamp ---
if date -v +1S +%s >/dev/null 2>&1; then
    # macOS
    WHOLE_HOURS="${HOURS%%.*}"
    FRAC=$(echo "$HOURS" | sed 's/^[0-9]*//' | sed 's/^\.//')
    if [ -n "$FRAC" ]; then
        MINUTES=$(echo "scale=0; 0.$FRAC * 60 / 1" | bc 2>/dev/null || echo "0")
    else
        MINUTES=0
    fi
    DEADLINE=$(date -v "+${WHOLE_HOURS:-0}H" -v "+${MINUTES}M" +%s)
    DEADLINE_DISPLAY=$(date -r "$DEADLINE" '+%H:%M %Z')
else
    # Linux
    DEADLINE=$(date -d "+${HOURS} hours" +%s)
    DEADLINE_DISPLAY=$(date -d "@$DEADLINE" '+%H:%M %Z')
fi

echo "=== lmxlab Autorun ==="
echo "Focus:      ${FOCUS}"
echo "Duration:   ${HOURS} hours"
echo "Deadline:   ${DEADLINE_DISPLAY}"
echo "Max turns:  ${MAX_TURNS}"
echo "Starting autonomous research loop..."
echo ""

# Allowed tools: python training, git (local only), web research,
# file operations, experiment tracking
AUTORUN_DEADLINE="$DEADLINE" claude -p \
    "Run /autorun ${FOCUS}. Research and run experiments for $HOURS hours (deadline: $DEADLINE_DISPLAY). Use soft planning (no EnterPlanMode). When you finish an iteration, immediately start the next one without waiting for input." \
    --allowedTools "Read,Edit,Write,Grep,Glob,Bash(uv run *),Bash(cd *parameter-golf* && *),Bash(python* *),Bash(git add *),Bash(git commit *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git checkout -- *),Bash(mkdir *),Bash(cp *),Bash(date *),Bash(ls *),Bash(wc *),WebSearch,WebFetch(*)" \
    --max-turns "$MAX_TURNS"
