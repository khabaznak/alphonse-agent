#!/bin/sh
set -eu

usage() {
  echo "Usage: $0 PID MODE [WARMUP_SECONDS] [DURATION_SECONDS] [INTERVAL_SECONDS] [OUTPUT_CSV]" >&2
  exit 2
}

[ "$#" -ge 2 ] || usage

target_pid=$1
diagnostic_mode=$2
warmup_seconds=${3:-60}
duration_seconds=${4:-300}
interval_seconds=${5:-30}
output_csv=${6:-"webkit-memory-${diagnostic_mode}-${target_pid}.csv"}

case "$target_pid:$warmup_seconds:$duration_seconds:$interval_seconds" in
  *[!0-9:]*|:*|*::*|*:)
    usage
    ;;
esac

case "$diagnostic_mode" in
  normal|static|ping-only|poll-no-commit|render-only|history-static|history-render|history-render-plain|history-render-memo|history-render-timeline-memo) ;;
  *)
    echo "Unknown diagnostic mode: $diagnostic_mode" >&2
    usage
    ;;
esac

if ! kill -0 "$target_pid" 2>/dev/null; then
  echo "Process $target_pid is not running or is not accessible." >&2
  exit 1
fi

process_command=$(ps -p "$target_pid" -o command=)
case "$process_command" in
  *com.apple.WebKit.WebContent*) ;;
  *)
    echo "Process $target_pid is not a WebKit WebContent process: $process_command" >&2
    exit 1
    ;;
esac

summary_file=$(mktemp "${TMPDIR:-/tmp}/alphonse-webkit-vmmap.XXXXXX")
trap 'rm -f "$summary_file"' EXIT HUP INT TERM

echo "timestamp,elapsed_seconds,mode,physical_footprint,peak_footprint,live_allocations,allocated_bytes,default_resident,default_dirty,default_swapped,webkit_resident,webkit_dirty,webkit_swapped" > "$output_csv"

echo "Warming up PID $target_pid in $diagnostic_mode mode for $warmup_seconds seconds..." >&2
sleep "$warmup_seconds"

started_at=$(date +%s)
sample_at=0
while [ "$sample_at" -le "$duration_seconds" ]; do
  if ! kill -0 "$target_pid" 2>/dev/null; then
    echo "Process $target_pid exited during sampling." >&2
    exit 1
  fi

  vmmap --summary "$target_pid" > "$summary_file"
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  metrics=$(awk '
    /^Physical footprint:/ { physical = $3 }
    /^Physical footprint \(peak\):/ { peak = $4 }
    /^MALLOC ZONE[[:space:]]/ { zones = 1; next }
    zones && /^DefaultMallocZone_/ {
      default_resident = $3; default_dirty = $4; default_swapped = $5
    }
    zones && /^WebKit Malloc_/ {
      webkit_resident = $4; webkit_dirty = $5; webkit_swapped = $6
    }
    zones && /^TOTAL[[:space:]]/ {
      live_allocations = $6; allocated_bytes = $7
    }
    END {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", physical, peak, live_allocations, allocated_bytes,
        default_resident, default_dirty, default_swapped, webkit_resident, webkit_dirty, webkit_swapped
    }
  ' "$summary_file")
  echo "$timestamp,$sample_at,$diagnostic_mode,$metrics" >> "$output_csv"
  echo "Recorded $diagnostic_mode sample at ${sample_at}s." >&2

  [ "$sample_at" -eq "$duration_seconds" ] && break
  remaining=$((duration_seconds - sample_at))
  wait_seconds=$interval_seconds
  [ "$remaining" -lt "$interval_seconds" ] && wait_seconds=$remaining
  sleep "$wait_seconds"
  sample_at=$((sample_at + wait_seconds))
done

echo "Memory samples written to $output_csv" >&2
