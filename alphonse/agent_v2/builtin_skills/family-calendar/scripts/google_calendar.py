"""Report the managed Calendar migration to callers of the retired CLI."""
import sys


if __name__ == '__main__':
    print('{"error":"calendar_legacy_script_retired"}', file=sys.stderr)
    sys.exit(1)
