"""Regression for invoking an isolated supplement from an older research checkout."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

# These run inside a packaged NGFix launch, which sets LONGEVAL_TEST_PACKAGE.
pytestmark = pytest.mark.skipif('LONGEVAL_TEST_PACKAGE' not in os.environ,
                                reason='requires a packaged NGFix launch')


def test_safe_path_uses_supplement_from_old_checkout(tmp_path):
    package = Path(os.environ['LONGEVAL_TEST_PACKAGE']).resolve()
    old = tmp_path / 'old-checkout'
    (old / 'research/longeval').mkdir(parents=True)
    (old / 'research/longeval/__init__.py').write_text('# Historical package without ngfix_screen\n')
    env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1')
    command = ['-m', 'research.longeval.ngfix_screen', 'package-verify', str(package)]
    broken = subprocess.run([sys.executable, *command], cwd=old, env=env, capture_output=True, text=True)
    assert broken.returncode != 0 and 'No module named research.longeval.ngfix_screen' in broken.stderr
    fixed = subprocess.run([sys.executable, '-P', *command], cwd=old, env=env, capture_output=True, text=True)
    assert fixed.returncode == 0, fixed.stderr
    assert 'packaged files' in fixed.stdout


def test_launch_script_enables_safe_path_before_first_package_import():
    package = Path(os.environ['LONGEVAL_TEST_PACKAGE']).resolve()
    script = (package / 'scripts/pace/longeval-ngfix.sbatch').read_text()
    first_import = next(line for line in script.splitlines() if 'ngfix_screen package-verify' in line)
    assert first_import.startswith('"$PYTHON" -P -m ')
    assert script.index(first_import) < script.index('ngfix_screen download')
