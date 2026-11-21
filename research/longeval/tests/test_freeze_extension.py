"""The extension freeze refuses an unfinished protocol and records what already existed."""
import hashlib
import json

import pytest

from research.longeval import freeze_extension


def test_freeze_refuses_until_the_protocol_is_final(tmp_path):
    protocol, receipt = tmp_path / 'protocol.md', tmp_path / 'receipt.json'
    (tmp_path / 'code.py').write_text('x = 1\n')
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


def test_the_draft_is_not_freezable_and_every_listed_file_exists():
    # The committed draft holds open decisions, and the final file does not exist yet.
    draft = freeze_extension.HERE / 'docs/protocol/EXTENSION_PROTOCOL_DRAFT.md'
    assert freeze_extension.OPEN in draft.read_text(encoding='utf-8')
    assert not freeze_extension.RECEIPT.exists()
    missing = [name for name in freeze_extension.FROZEN[1:] if not (freeze_extension.ROOT / name).is_file()]
    assert missing == []
