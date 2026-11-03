"""No real processes/signals: deadline and logging contracts with owned-child doubles."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from gen.record_command import main


class RecordCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='record-command-')
        self.addCleanup(temporary.cleanup)
        self.prefix = Path(temporary.name) / 'attempt'
        self.deadline = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        self.child = Mock(pid=123, returncode=0)
        self.child.poll.return_value = 0
        self.child.wait.return_value = 0
        self.args = ['--record', str(self.prefix), '--deadline', self.deadline, '--', 'example', '--flag']

    def metadata(self):
        return json.loads(Path(str(self.prefix) + '.json').read_text())

    def test_success_records_exact_argv_and_refuses_overwrite(self):
        with patch('gen.record_command.subprocess.Popen', return_value=self.child) as spawn:
            self.assertEqual(main(self.args), 0)
            self.assertEqual(spawn.call_args.args[0], ['example', '--flag'])
            self.assertNotIn('shell', spawn.call_args.kwargs)
            with self.assertRaises(SystemExit):
                main(self.args)
        self.assertEqual(self.metadata()['status'], 'completed')
        self.assertEqual(self.metadata()['exit_code'], 0)
        self.assertIsNotNone(self.metadata()['log_sha256'])

    def test_deadline_interrupts_owned_child(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired(['example'], 1), 0]
        self.child.returncode = -signal.SIGINT
        with patch('gen.record_command.subprocess.Popen', return_value=self.child):
            self.assertEqual(main(self.args), 1)
        self.child.send_signal.assert_called_once_with(signal.SIGINT)
        self.child.kill.assert_not_called()
        self.assertEqual(self.metadata()['status'], 'paused_deadline')

    def test_no_deadline_waits_without_timeout(self):
        args = ['--record', str(self.prefix), '--', 'example', '--flag']
        with patch('gen.record_command.subprocess.Popen', return_value=self.child):
            self.assertEqual(main(args), 0)
        self.child.wait.assert_called_once_with(timeout=None)
        self.child.send_signal.assert_not_called()
        self.assertIsNone(self.metadata()['deadline_utc'])
        self.assertEqual(self.metadata()['status'], 'completed')

    def test_no_deadline_still_honors_explicit_interruption(self):
        args = ['--record', str(self.prefix), '--', 'example']
        self.child.wait.side_effect = [KeyboardInterrupt(), 0]
        self.child.returncode = -signal.SIGINT
        with patch('gen.record_command.subprocess.Popen', return_value=self.child):
            self.assertEqual(main(args), 1)
        self.child.send_signal.assert_called_once_with(signal.SIGINT)
        self.child.kill.assert_not_called()
        self.assertEqual(self.metadata()['status'], 'paused_interrupt')

    def test_unresponsive_child_is_killed_after_grace(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired(['example'], 1),
                                       subprocess.TimeoutExpired(['example'], 10), 0]
        self.child.returncode = -signal.SIGKILL
        with patch('gen.record_command.subprocess.Popen', return_value=self.child):
            self.assertEqual(main(self.args), 1)
        self.child.kill.assert_called_once()
        self.assertTrue(self.metadata()['forced_kill'])

    def test_expired_deadline_never_starts_child(self):
        args = ['--record', str(self.prefix), '--deadline', '2000-01-01T00:00:00Z', '--', 'example']
        with patch('gen.record_command.subprocess.Popen') as spawn:
            with self.assertRaises(SystemExit):
                main(args)
        spawn.assert_not_called()
        self.assertFalse(Path(str(self.prefix) + '.json').exists())

    def test_spawn_failure_is_recorded_as_failure(self):
        with patch('gen.record_command.subprocess.Popen', side_effect=OSError('missing executable')):
            with self.assertRaises(OSError):
                main(self.args)
        self.assertEqual(self.metadata()['status'], 'failed')
        self.assertIn('missing executable', self.metadata()['error'])


if __name__ == '__main__':
    unittest.main()
