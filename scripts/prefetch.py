import argparse
import time

import mteb
from huggingface_hub import snapshot_download


def fetch(repo_id, revision, repo_type="model", attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(repo_id, revision=revision, repo_type=repo_type)
            print(f"ok {repo_type} {repo_id}@{revision} -> {path}")
            return
        except Exception as exc:
            print(f"attempt {attempt}/{attempts} failed for {repo_id}: {type(exc).__name__}: {exc}")
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"could not download {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Download models and the AppsRetrieval dataset before evaluation")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.models:
        meta = mteb.get_model_meta(name)
        fetch(name, meta.revision)
    dataset = mteb.get_task("AppsRetrieval").metadata.dataset
    fetch(dataset["path"], dataset["revision"], repo_type="dataset")


if __name__ == "__main__":
    main()
