import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="LongEval research pipeline foundation")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="create deterministic synthetic data in a new directory")
    fixture.add_argument("output", type=Path)
    run = sub.add_parser("run", help="run a three-snapshot development pilot")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--native-origin", choices=("checkout-build", "released-wheel", "unverified"), default="unverified",
                     help="record actual binary origin; this declaration is not a source-build verification")
    verify = sub.add_parser("verify", help="verify checksums and optionally re-score relevance metrics")
    verify.add_argument("output", type=Path)
    verify.add_argument("--dataset", type=Path)
    fetch = sub.add_parser("fetch-release", help="download only pinned real query/qrel files (bounded size)")
    fetch.add_argument("folder", type=Path)
    audit = sub.add_parser("audit-release", help="offline checksum, judgment-cohort and query-role audit")
    audit.add_argument("folder", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    dense = sub.add_parser("inspect-dense", help="inspect bounded archive prefixes; no full vector downloads")
    dense.add_argument("folder", type=Path)
    dense.add_argument("--output", type=Path, required=True)
    dense.add_argument("--offline", action="store_true")
    dense.add_argument("--release-files", type=Path, help="also check qrel document coverage against stored IDs")
    ingest = sub.add_parser("ingest-dense", help="resume full pinned downloads and stream-extract verified archives")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--snapshots", type=int, nargs="+", default=[1, 2, 3])
    vectors = sub.add_parser("audit-vectors", help="scan all stored vectors and join shared IDs across snapshots")
    vectors.add_argument("folder", type=Path)
    vectors.add_argument("--output", type=Path, required=True)
    vectors.add_argument("--snapshots", type=int, nargs="+", default=[1, 2, 3])
    vectors.add_argument("--chunk-rows", type=int, default=2048)
    prepare = sub.add_parser("prepare-dense", help="resume all dense downloads, extract, and audit every vector")
    prepare.add_argument("folder", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    resource = sub.add_parser("feasibility", help="run one backend on a subset of audited vectors with held-out document probes")
    resource.add_argument("folder", type=Path)
    resource.add_argument("--audit", type=Path, required=True)
    resource.add_argument("--output", type=Path, required=True)
    resource.add_argument("--snapshot", type=int, default=1)
    resource.add_argument("--backend", choices=("epzero", "faiss_hnsw", "faiss_ivf"), default="epzero")
    resource.add_argument("--count", type=int, default=8000)
    resource.add_argument("--native-origin", choices=("checkout-build", "released-wheel", "unverified"), default="unverified")
    docs = sub.add_parser("fetch-documents", help="resume pinned original document ZIP downloads")
    docs.add_argument("manifest", type=Path)
    docs.add_argument("folder", type=Path)
    texts = sub.add_parser("audit-documents", help="audit all eligible title/abstract text and shared IDs")
    texts.add_argument("manifest", type=Path)
    texts.add_argument("folder", type=Path)
    texts.add_argument("--output", type=Path, required=True)
    texts.add_argument("--snapshots", type=int, nargs="+", default=[1, 2, 3])
    texts.add_argument("--dense-maps", type=Path)
    model = sub.add_parser("fetch-encoder", help="fetch and verify revision-pinned model/tokenizer files")
    model.add_argument("folder", type=Path)
    encode = sub.add_parser("encode-documents", help="offline document encoder compatibility diagnostic")
    encode.add_argument("model", type=Path)
    encode.add_argument("records", type=Path)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    encode.add_argument("--max-length", type=int, default=8192)
    encode.add_argument("--threads", type=int, default=4)
    encode.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "fixture":
        from .fixture import make_fixture
        print(make_fixture(args.output))
    elif args.command == "run":
        from .runner import run_pilot
        manifest = run_pilot(args.dataset, args.config, args.output, args.native_origin)
        print(f"{manifest['status']}: {manifest['result_rows']} result rows; synthetic={manifest['synthetic']}")
    elif args.command == "verify":
        from .runner import verify_artifacts
        manifest = verify_artifacts(args.output, args.dataset)
        print(f"Verified {len(manifest['artifacts'])} artifacts; synthetic={manifest['synthetic']}")
    elif args.command == "fetch-release":
        from .release import fetch_release
        print(f"Verified {fetch_release(args.folder)} released files")
    elif args.command == "audit-release":
        from .release import audit_release
        report = audit_release(args.folder, args.output)
        print(f"Audited {report['queries']['train']} train and {report['queries']['test']} test queries")
        print({name: panel['count'] for name, panel in report['retrospective_panels'].items()})
    elif args.command == "inspect-dense":
        from .dense import inspect_dense
        report = inspect_dense(args.folder, args.output, offline=args.offline, release_files=args.release_files)
        print([item["metadata"] for item in report["artifacts"]])
    elif args.command == "ingest-dense":
        from .storage import ingest_dense
        print(f"Verified {len(ingest_dense(args.folder, args.snapshots))} full dense archives")
    elif args.command == "audit-vectors":
        from .mapped import audit_vectors
        report = audit_vectors(args.folder, args.output, snapshots=args.snapshots, chunk_rows=args.chunk_rows)
        print(f"Audited {len(report['snapshots'])} snapshots; full fixed-vector gate={report['fixed_vector_gate_passed']}")
    elif args.command == "prepare-dense":
        from .storage import prepare_dense
        report = prepare_dense(args.folder, args.output)
        print(f"Preparation {report['status']}; full fixed-vector gate={report['fixed_vector_gate_passed']}")
    elif args.command == "feasibility":
        from .feasibility import run_feasibility
        report = run_feasibility(args.folder, args.audit, args.output, snapshot=args.snapshot, kind=args.backend,
                                 count=args.count, native_origin=args.native_origin)
        print(f"Feasibility {report['status']}: {report['backend']}, {report['count']} vectors; not a relevance experiment")
    elif args.command == "fetch-documents":
        from .documents import fetch_documents
        print(f"Verified {len(fetch_documents(args.manifest, args.folder))} document archives")
    elif args.command == "audit-documents":
        from .documents import audit_documents
        report = audit_documents(args.manifest, args.folder, args.output, snapshots=args.snapshots, dense_maps=args.dense_maps)
        print(f"Text audit {report['status']}; full fixed-text gate={report['fixed_text_gate_passed']}")
    elif args.command == "fetch-encoder":
        from .encoder import fetch_model
        print(f"Verified encoder revision {fetch_model(args.folder)['revision']}")
    elif args.command == "encode-documents":
        from .encoder import encode_documents
        report = encode_documents(args.model, args.records, args.output, dtype=args.dtype, max_length=args.max_length, threads=args.threads, device=args.device)
        print(f"Document encoder diagnostic {report['status']}; not research query embeddings")


if __name__ == "__main__":
    main()
