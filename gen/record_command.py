"""Run one foreground command with a durable log and an optional UTC deadline.

Use a new record prefix for every attempt. Only this command's child is signalled;
this does not inspect or terminate unrelated jobs. No shell interpretation.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from age.common import save_json, sha256
from .quality_experiment import parse_deadline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True, help='New prefix for .json and .log artifacts')
    parser.add_argument('--deadline', help='Explicit one-shot ISO time; omitted means no time limit')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command after -- is required')
    deadline = parse_deadline(args.deadline)
    now = lambda: datetime.now(timezone.utc)
    if deadline is not None and deadline <= now().timestamp() + 20:
        parser.error('Need at least 20 seconds before the deadline, including shutdown reserve')
    record, log = Path(str(args.record) + '.json'), Path(str(args.record) + '.log')
    if record.exists() or log.exists():
        parser.error('Record prefix already exists; never overwrite an earlier attempt')
    record.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(command=command, cwd=str(Path.cwd()), start_utc=now().isoformat(),
                    deadline_utc=args.deadline, status='running', exit_code=None,
                    note='No automatic time limit when deadline_utc is null. An explicit deadline or interruption '
                         'sends SIGINT, then kills this direct child after a 10-second grace period.')
    save_json(record, metadata)
    print('Starting recorded command:', command, flush=True)
    child = None
    try:
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                                              TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1'))
            metadata['pid'] = child.pid
            save_json(record, metadata)
            try:
                timeout = None if deadline is None else max(0.01, deadline - now().timestamp() - 15)
                child.wait(timeout=timeout)
                metadata['status'] = 'completed' if child.returncode == 0 else 'failed'
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                metadata['status'] = 'paused_deadline' if isinstance(exc, subprocess.TimeoutExpired) else 'paused_interrupt'
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    metadata['forced_kill'] = True
            metadata['exit_code'] = child.returncode
    except BaseException as exc:
        metadata.update(status='paused_interrupt' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        metadata['end_utc'] = now().isoformat()
        metadata['log_sha256'] = sha256(log) if log.exists() else None
        save_json(record, metadata)
        print(f'Recorded {metadata["status"]}: {record}', flush=True)
    return 0 if metadata['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
