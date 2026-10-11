"""An OOM without a killed process must still fail resource qualification."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.vln import run_episodic_gate as gate


class FinishedCommand:
    pid = 12345
    def poll(self):
        return 0
    def wait(self, **kwargs):
        return 0


class SupervisorOomTest(unittest.TestCase):
    def run_supervisor(self, oom_after, transient=False):
        sample = dict(job_limit_gib=512, job_usage_gib=10, working_set_gib=10,
                      anon_gib=10, job_peak_gib=10, oom_kill=0, oom=0, limit_events=0)
        command = FinishedCommand()
        if transient:
            polls = iter([None, 0])
            command.poll = lambda: next(polls)
        samples = [sample, dict(sample, oom=1), sample] if transient else [sample, dict(sample, oom=oom_after)]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['gate', '--out', directory, '--', 'true']), \
                 patch.object(gate, 'job_cgroup', return_value=(2, Path('/fake/job_1'))), \
                 patch.object(gate, 'cgroup_sample', side_effect=samples), \
                 patch.object(gate.subprocess, 'Popen', return_value=command), \
                 patch.object(gate.os, 'getpgid', return_value=12345), \
                 patch.object(gate.os, 'killpg') as kill, \
                 patch.object(gate, 'gpu_sample', return_value=[]), \
                 self.assertRaises(SystemExit) as result:
                gate.main()
            if transient:
                self.assertTrue(kill.called)
            report = json.loads((Path(directory) / 'resources.json').read_text())
        return result.exception.code, report

    def test_recoverable_oom_fails_even_when_command_exits_zero(self):
        code, report = self.run_supervisor(1)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['new_oom_events'], 1)

    def test_observed_transient_oom_stops_owned_command(self):
        code, report = self.run_supervisor(0, transient=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['new_oom_events'], 1)

    def test_clean_command_still_passes(self):
        code, report = self.run_supervisor(0)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
