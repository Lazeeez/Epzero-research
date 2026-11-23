"""The extension freeze refuses an unfinished protocol and records what already existed."""
import hashlib
import json

import pytest

from research.longeval import freeze_extension


def test_freeze_refuses_until_the_protocol_is_final(tmp_path):
    protocol, receipt = tmp_path / 'protocol.md', tmp_path / 'receipt.json'
    (tmp_path / 'code.py').write_bytes(b'x = 1\n')                 # bytes: the hash must not depend on the platform
    (tmp_path / 'build/pilot').mkdir(parents=True)
    arguments = dict(root=tmp_path, protocol=protocol, receipt=receipt, files=('protocol.md', 'code.py'),
                     outcomes={'pilot': 'build/pilot', 'audit': 'build/audit'})
    with pytest.raises(FileNotFoundError):
        freeze_extension.freeze(**arguments)
    protocol.write_text('Sampling: **TO DECIDE**. Pool: **TO DECIDE**.\n')
    with pytest.raises(ValueError, match='2 open decisions'):
        freeze_extension.freeze(**arguments)
    assert not receipt.exists()
    protocol.write_text('Sampling: temperature 0. Pool: primary arms.\n')
    record = freeze_extension.freeze(**arguments)
    assert record['files'] == {'protocol.md': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                               'code.py': hashlib.sha256(b'x = 1\n').hexdigest()}
    assert record['outcomes_present_at_freeze'] == {'pilot': True, 'audit': False}
    assert json.loads(receipt.read_text())['protocol'] == 'EXTENSION_PROTOCOL_V1'
    with pytest.raises(FileExistsError):
        freeze_extension.freeze(**arguments)


def test_the_protocol_has_no_open_decision_and_every_listed_file_exists():
    # Every decision is recorded (23 November 2026), in the draft or in the finalised file.
    final = freeze_extension.PROTOCOL
    draft = freeze_extension.HERE / 'docs/protocol/EXTENSION_PROTOCOL_DRAFT.md'
    text = (final if final.is_file() else draft).read_text(encoding='utf-8')
    assert freeze_extension.OPEN not in text
    missing = [name for name in freeze_extension.FROZEN[1:] if not (freeze_extension.ROOT / name).is_file()]
    assert missing == []


def test_a_changed_file_blocks_the_freeze(tmp_path):
    import shutil
    import subprocess
    if shutil.which('git') is None:
        pytest.skip('needs git')
    git = lambda *arguments: subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *arguments],
                                            cwd=tmp_path, check=True, capture_output=True)
    protocol, scorer = tmp_path / 'protocol.md', tmp_path / 'scorer.py'
    protocol.write_bytes(b'Everything is decided.\n')
    scorer.write_bytes(b'MARGIN = 1\n')
    git('init', '-q')
    git('add', 'protocol.md', 'scorer.py')
    git('commit', '-q', '-m', 'design')
    arguments = dict(root=tmp_path, protocol=protocol, receipt=tmp_path / 'receipt.json', files=('protocol.md', 'scorer.py'),
                     outcomes={})
    # The scoring module differs from the committed one: nothing may be frozen.
    scorer.write_bytes(b'MARGIN = 2\n')
    with pytest.raises(ValueError, match='uncommitted changes'):
        freeze_extension.freeze(**arguments)
    assert not (tmp_path / 'receipt.json').exists()
    git('commit', '-q', '-am', 'change the margin')
    record = freeze_extension.freeze(**arguments)
    assert record['git_commit'] and record['files']['scorer.py'] == hashlib.sha256(b'MARGIN = 2\n').hexdigest()
