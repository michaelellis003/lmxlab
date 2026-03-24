#!/usr/bin/env bash
# Tmux watcher that re-injects a prompt when Claude Code goes idle.
#
# Sets up a tmux session with Claude Code, then monitors it.
# When Claude stops (waiting for input), the watcher re-injects
# the prompt to keep it going.
#
# Usage:
#   ./scripts/watcher.sh <prompt> [options]
#
# Examples:
#   ./scripts/watcher.sh "/autorun HYP-008"
#   ./scripts/watcher.sh "/autorun HYP-008" --max 5
#   ./scripts/watcher.sh "/autorun HYP-008" --cooldown 30
#   ./scripts/watcher.sh "/autorun HYP-008" --session myresearch
#
# What happens:
#   1. Creates a tmux session running `claude`
#   2. Attaches you to it in one pane
#   3. Monitors from a background process
#   4. When Claude is idle, sends the prompt as keystrokes
#
# Stop: Ctrl+C in the watcher, or `tmux kill-session -t <name>`

set -euo pipefail

TMUX_BIN="$(command -v tmux || echo /opt/homebrew/bin/tmux)"

# --- Defaults ---
SESSION="claude-autorun"
INTERVAL=5         # seconds between polls
COOLDOWN=15        # seconds of confirmed idle before injection
MAX_INJECTIONS=0   # 0 = unlimited
STARTUP_WAIT=10    # seconds to wait before first poll
HOURS=0            # 0 = no time limit
DEADLINE=0         # computed from HOURS
PERSONALITY=""     # extracted from prompt for nudge messages

# --- Parse args ---
PROMPT="${1:?Usage: watcher.sh <prompt> [--hours N] [--max N] [--cooldown N] [--interval N] [--session NAME]}"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours)     HOURS="$2"; shift 2 ;;
        --max)       MAX_INJECTIONS="$2"; shift 2 ;;
        --cooldown)  COOLDOWN="$2"; shift 2 ;;
        --interval)  INTERVAL="$2"; shift 2 ;;
        --session)   SESSION="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Extract personality from prompt ---
# If prompt is "/autorun pgolf ..." extract "pgolf" as personality
if [[ "$PROMPT" =~ ^/autorun[[:space:]]+([a-z]+) ]]; then
    PERSONALITY="${BASH_REMATCH[1]}"
fi

# --- Build nudge message ---
# On re-injection, remind the agent of its personality and goals
# instead of just re-sending the raw slash command.
build_nudge() {
    local count="$1"
    if [[ -n "$PERSONALITY" ]]; then
        cat <<EOF
Re-read your personality file (.claude/skills/autorun/personalities/${PERSONALITY}.md). Remember your identity, your cross-disciplinary research approach, and your goals. Review your memory files to see where you left off. Then decide: what is the single highest-value next step you can take right now? Run ${PROMPT}
EOF
    else
        echo "$PROMPT"
    fi
}

# --- Compute deadline from --hours ---
if [[ "$HOURS" != "0" ]]; then
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
    else
        # Linux
        DEADLINE=$(date -d "+${HOURS} hours" +%s)
    fi
fi

# --- Helpers ---
log() {
    echo "[watcher $(date '+%H:%M:%S')] $*"
}

capture_pane() {
    "$TMUX_BIN" capture-pane -t "${SESSION}:0" -p 2>/dev/null || echo ""
}

# Idle detection: two signals must agree
#   1. No "working" indicator visible (spinner, tool execution)
#   2. Pane content stable for STABLE_THRESHOLD polls
PREV_HASH=""
STABLE_COUNT=0
STABLE_THRESHOLD=3  # unchanged for 3 polls = stable

is_idle() {
    local content
    content=$(capture_pane)

    # Empty = session not ready
    [[ -z "$content" ]] && return 1

    # Claude is WORKING if the status line shows activity.
    # "Esc to interrupt" appears while Claude is thinking/executing.
    if echo "$content" | grep -qi 'esc to interrupt' 2>/dev/null; then
        STABLE_COUNT=0
        PREV_HASH=""
        return 1
    fi

    # Content stability check using hash (avoids issues with
    # large string comparisons and invisible character diffs)
    local hash
    hash=$(echo "$content" | md5 -q 2>/dev/null \
        || echo "$content" | md5sum 2>/dev/null | cut -d' ' -f1)

    if [[ "$hash" == "$PREV_HASH" ]]; then
        STABLE_COUNT=$((STABLE_COUNT + 1))
    else
        STABLE_COUNT=0
        PREV_HASH="$hash"
    fi

    [[ $STABLE_COUNT -ge $STABLE_THRESHOLD ]]
}

# --- Check for existing session ---
if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    log "Session '$SESSION' already exists. Attaching watcher."
else
    log "Creating tmux session '$SESSION' with claude..."
    "$TMUX_BIN" new-session -d -s "$SESSION" -x 200 -y 50 \
        "claude --dangerously-skip-permissions"
    log "Waiting ${STARTUP_WAIT}s for Claude to start..."
    sleep "$STARTUP_WAIT"
fi

# --- Banner ---
MAX_DISPLAY="unlimited"
if [[ $MAX_INJECTIONS -gt 0 ]]; then
    MAX_DISPLAY="$MAX_INJECTIONS"
fi

DEADLINE_DISPLAY="none"
if [[ $DEADLINE -gt 0 ]]; then
    if date -r "$DEADLINE" '+%H:%M %Z' >/dev/null 2>&1; then
        DEADLINE_DISPLAY=$(date -r "$DEADLINE" '+%H:%M %Z')
    else
        DEADLINE_DISPLAY=$(date -d "@$DEADLINE" '+%H:%M %Z')
    fi
fi

echo "╔══════════════════════════════════════╗"
echo "║          Claude Code Watcher         ║"
echo "╠══════════════════════════════════════╣"
echo "║  Session:  ${SESSION}"
echo "║  Prompt:   ${PROMPT}"
echo "║  Interval: ${INTERVAL}s"
echo "║  Cooldown: ${COOLDOWN}s"
echo "║  Max:      ${MAX_DISPLAY}"
echo "║  Deadline: ${DEADLINE_DISPLAY}"
echo "╚══════════════════════════════════════╝"
echo ""
log "Monitoring session '${SESSION}'..."
log "Attach with: tmux attach -t ${SESSION}"
echo ""

trap 'log "Stopped."; exit 0' INT TERM

injection_count=0
idle_since=0

while true; do
    # Check session still exists
    if ! "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
        log "Session '$SESSION' is gone. Exiting."
        exit 1
    fi

    # Check time limit
    if [[ $DEADLINE -gt 0 ]] && [[ $(date +%s) -ge $DEADLINE ]]; then
        log "Time limit reached (${HOURS}h). Exiting watcher."
        log "Session '${SESSION}' is still running — attach to review."
        exit 0
    fi

    if is_idle; then
        if [[ $idle_since -eq 0 ]]; then
            idle_since=$(date +%s)
            log "Idle detected (stable ${STABLE_COUNT}x). Waiting ${COOLDOWN}s..."
        fi

        now=$(date +%s)
        elapsed=$((now - idle_since))

        if [[ $elapsed -ge $COOLDOWN ]]; then
            injection_count=$((injection_count + 1))
            NUDGE=$(build_nudge "$injection_count")
            log "Injecting prompt (#${injection_count}): ${PROMPT}"

            # Clear any pending input, then send nudge
            "$TMUX_BIN" send-keys -t "${SESSION}:0" C-c
            sleep 1
            "$TMUX_BIN" send-keys -t "${SESSION}:0" C-u
            sleep 1
            "$TMUX_BIN" send-keys -t "${SESSION}:0" "$NUDGE" Enter

            # Verify Claude started working (retry Enter if not)
            log "Verifying submission..."
            for attempt in 1 2 3; do
                sleep 5
                verify_content=$("$TMUX_BIN" capture-pane \
                    -t "${SESSION}:0" -p 2>/dev/null || echo "")
                if echo "$verify_content" \
                    | grep -qi 'esc to interrupt' 2>/dev/null; then
                    log "Confirmed: Claude is working."
                    break
                fi
                if [[ $attempt -lt 3 ]]; then
                    log "Not started yet (attempt ${attempt}/3). Sending Enter..."
                    "$TMUX_BIN" send-keys -t "${SESSION}:0" Enter
                else
                    log "Warning: Claude may not have started. Will retry next cycle."
                fi
            done

            idle_since=0
            STABLE_COUNT=0
            PREV_HASH=""

            # Check limit
            if [[ $MAX_INJECTIONS -gt 0 ]] \
                && [[ $injection_count -ge $MAX_INJECTIONS ]]; then
                log "Reached max ($MAX_INJECTIONS). Exiting watcher."
                log "Session '${SESSION}' is still running."
                exit 0
            fi

            # Wait for Claude to finish starting up
            log "Waiting 20s before resuming polls..."
            sleep 20
            continue
        fi
    else
        if [[ $idle_since -gt 0 ]]; then
            log "Active again. Reset."
        fi
        idle_since=0
    fi

    sleep "$INTERVAL"
done
