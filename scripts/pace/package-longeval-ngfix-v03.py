"""Derive a launcher-only correction from the immutable, compile-checked v02 package."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path.cwd()
NAME = 'longeval-ngfix-screen-03'
PARENT = 'longeval-ngfix-screen-02'
PARENT_SHA = '5532d5b8a7954efe33a291e1a583cfccaa743a9831f6aaaa859670fed66661f6'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    parent = ROOT / 'build' / (PARENT + '.tar.gz')
    assert sha(parent.read_bytes()) == PARENT_SHA
    with tarfile.open(parent) as archive:
        payload = {}
        for member in archive:
            assert member.isfile() and member.name.startswith(PARENT + '/')
            name = member.name[len(PARENT)+1:]
            assert name not in payload
            payload[name] = archive.extractfile(member).read()
    prior = json.loads(payload.pop('manifest.json'))
    assert {n: sha(raw) for n, raw in payload.items()} == prior['files']
    script_name = 'scripts/pace/longeval-ngfix.sbatch'
    script = payload[script_name].decode()
    assert script.count('PACKAGE="$ROOT/longeval-ngfix-screen-02"') == 1
    script = script.replace('PACKAGE="$ROOT/longeval-ngfix-screen-02"', 'PACKAGE="$ROOT/longeval-ngfix-screen-03"')
    script = script.replace('"$PYTHON" -m ', '"$PYTHON" -P -m ')
    script = script.replace('OUTPUT="$ROOT/build/pace-longeval-ngfix/jobs/${SLURM_JOB_ID:?requires Slurm}"',
        'OUTPUT="$ROOT/build/pace-longeval-ngfix/jobs/${SLURM_JOB_ID:?requires Slurm}"\nPHASE=initialization')
    script = script.replace('    code=$?\n', '    code=$?\n    printf \'exit_code=%s phase=%s\\n\' "$code" "$PHASE"\n')
    script = script.replace('module purge', 'PHASE=environment\necho "phase=$PHASE"\nmodule purge')
    script = script.replace('export PYTHONPATH="$PACKAGE"\n"$PYTHON" -P -m research.longeval.ngfix_screen package-verify',
        'export PYTHONPATH="$PACKAGE"\nPHASE=package_verification\necho "phase=$PHASE"\n"$PYTHON" -P -m research.longeval.ngfix_screen package-verify', 1)
    script = script.replace('g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 \\\n',
        'PHASE=compile\necho "phase=$PHASE"\ng++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 \\\n', 1)
    script = script.replace('srun --cpu-bind=cores "$PYTHON" -P -m pytest',
        'PHASE=tests\necho "phase=$PHASE"\nexport LONGEVAL_TEST_PACKAGE="$PACKAGE"\nsrun --cpu-bind=cores "$PYTHON" -P -m pytest', 1)
    script = script.replace('research/longeval/tests/test_ngfix_screen.py \\\n',
        'research/longeval/tests/test_ngfix_screen.py scripts/pace/tests/test_ngfix_launch.py \\\n', 1)
    script = script.replace('"$PYTHON" -P -m research.longeval.ngfix_screen fixture',
        'PHASE=native_fixture\necho "phase=$PHASE"\n"$PYTHON" -P -m research.longeval.ngfix_screen fixture', 1)
    script = script.replace('"$PYTHON" -P -m research.longeval.ngfix_screen download',
        'PHASE=download\necho "phase=$PHASE"\n"$PYTHON" -P -m research.longeval.ngfix_screen download', 1)
    script = script.replace('srun --cpu-bind=cores "$PYTHON" -P -m research.longeval.ngfix_screen sample',
        'PHASE=sample\necho "phase=$PHASE"\nsrun --cpu-bind=cores "$PYTHON" -P -m research.longeval.ngfix_screen sample', 1)
    payload[script_name] = script.encode()
    doc = payload['scripts/pace/LONGEVAL_NGFIX.md'].decode().replace(PARENT, NAME)
    doc = doc.replace('-m research.longeval.ngfix_screen package-verify', '-P -m research.longeval.ngfix_screen package-verify', 1)
    doc = doc.replace('Use only version02.', 'Version02 also has a launch-path shadowing bug when invoked from the old checkout. Use version03; it enables Python safe-path mode (-P). Native algorithm, input files and compiler flags are unchanged.')
    payload['scripts/pace/LONGEVAL_NGFIX.md'] = doc.encode()
    for name in ('scripts/pace/tests/test_ngfix_launch.py', 'scripts/pace/package-longeval-ngfix-v03.py'):
        payload[name] = (ROOT / name).read_bytes()
    changed = [n for n in prior['files'] if sha(payload[n]) != prior['files'][n]]
    assert set(changed) == {script_name, 'scripts/pace/LONGEVAL_NGFIX.md'}
    manifest = {**prior, 'parent_archive_sha256': PARENT_SHA,
        'correction': 'Python -P prevents the old checkout from shadowing the isolated supplement; phase diagnostics added',
        'changed_parent_files': changed, 'files': {n: sha(raw) for n, raw in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    check = ROOT / 'build' / (NAME+'-check')
    check.mkdir(exist_ok=False)
    for name, raw in payload.items():
        path = check / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    destination = ROOT / 'build' / (NAME+'.tar.gz')
    with destination.open('xb') as target, gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(NAME+'/'+name); info.size = len(raw)
                info.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(destination) as archive:
        assert {m.name for m in archive} == {NAME+'/'+n for n in payload}
        for name, raw in payload.items(): assert archive.extractfile(NAME+'/'+name).read() == raw
    receipt = {'archive': destination.name, 'sha256': sha(destination.read_bytes()), 'bytes': destination.stat().st_size,
        'files': len(manifest['files']), 'parent_sha256': PARENT_SHA, 'changed_parent_files': changed,
        'native_runtime_validated_locally': False}
    (ROOT / 'build' / (NAME+'-receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
