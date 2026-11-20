"""Storage-routing checks only: no model/package downloads or scheduler calls."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

HELPER = Path(__file__).resolve().parents[1] / 'longeval-node-env.sh'
SETUP = HELPER.with_name('longeval-gpu-setup.sh')
KEYS = ('LONGEVAL_NODE_STAGE', 'TMPDIR', 'TMP', 'TEMP', 'PIP_CACHE_DIR', 'UV_CACHE_DIR',
        'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_BIN_DIR', 'HF_HOME', 'HF_HUB_CACHE',
        'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH',
        'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')


@unittest.skipUnless(sys.platform.startswith('linux'), 'the helper is a shell script for Linux compute nodes')
class NodeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='longeval-node-env-', dir='/tmp')
        self.root = Path(self.temp.name)
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.mount_type('ext4')
        self.env = dict(os.environ)
        for key in (*KEYS, 'SLURM_TMPDIR', 'LONGEVAL_NODE_TMP_ROOT', 'SLURM_JOB_ID', 'SLURMD_NODENAME'):
            self.env.pop(key, None)
        self.env.update(TMPDIR=str(self.root), LONGEVAL_NODE_STAGE=str(self.stage),
                        SLURM_JOB_ID='test123', SLURMD_NODENAME='test-compute',
                        PATH=str(self.bin) + ':' + self.env['PATH'])

    def tearDown(self):
        self.temp.cleanup()

    def mount_type(self, kind):
        script = self.bin / 'findmnt'
        script.write_text('#!/bin/sh\nprintf "%s\\n" ' + shlex.quote(kind) + '\n')
        script.chmod(0o755)

    def invoke(self, extra=''):
        command = 'set -euo pipefail; source "$1"; ' + extra
        return subprocess.run(['bash', '-c', command, 'test', str(HELPER)],
                              env=self.env, capture_output=True, text=True)

    def test_all_download_and_cache_paths_stay_under_node_stage(self):
        script = 'import os,json; print(json.dumps({k:os.environ[k] for k in ' + repr(KEYS) + '}))'
        result = self.invoke('/usr/bin/python3 -c "' + script.replace('"', '\\"') + '"')
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        for key, value in values.items():
            path = Path(value).resolve()
            self.assertTrue(path == self.stage or self.stage in path.parents, key)

    def test_rejects_shared_and_memory_filesystems(self):
        for kind in ('nfs4', 'gpfs', 'lustre', 'tmpfs'):
            with self.subTest(kind=kind):
                self.mount_type(kind)
                result = self.invoke('echo SHOULD_NOT_RUN')
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('SHOULD_NOT_RUN', result.stdout)
                self.assertFalse((self.stage / 'tmp').exists())

    def test_accepts_repeated_xfs_records_from_pace(self):
        self.mount_type('xfs\nxfs')
        result = self.invoke('source "$1"; echo ACCEPTED')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ACCEPTED', result.stdout)
        self.assertTrue((self.stage / 'tmp').is_dir())

    def test_rejects_mixed_local_and_shared_records(self):
        for output in ('xfs\nnfs4', 'gpfs\nxfs', 'xfs\nxfs\nlustre'):
            with self.subTest(output=output):
                self.mount_type(output)
                result = self.invoke('echo SHOULD_NOT_RUN')
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('SHOULD_NOT_RUN', result.stdout)

    def test_rejects_empty_filesystem_result(self):
        self.mount_type('')
        result = self.invoke('echo SHOULD_NOT_RUN')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('SHOULD_NOT_RUN', result.stdout)

    def test_rejects_failed_filesystem_lookup(self):
        (self.bin / 'findmnt').write_text('#!/bin/sh\nexit 1\n')
        result = self.invoke('echo SHOULD_NOT_RUN')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('SHOULD_NOT_RUN', result.stdout)

    def test_rejects_stage_outside_temporary_root(self):
        self.env['LONGEVAL_NODE_STAGE'] = '/tmp'
        result = self.invoke('echo SHOULD_NOT_RUN')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be inside', result.stderr)

    def test_login_setup_fails_before_installation(self):
        self.env.pop('SLURM_JOB_ID')
        self.env.pop('SLURMD_NODENAME')
        result = subprocess.run(['bash', str(SETUP)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires a Slurm compute job', result.stderr)
        self.assertFalse((self.stage / 'venv').exists())


if __name__ == '__main__':
    unittest.main()
