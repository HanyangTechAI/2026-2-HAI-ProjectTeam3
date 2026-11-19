"""Run the separately installed official FADING code with local weights."""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--gender", choices=["female", "male"], required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--self-replace", type=float, default=0.9)
    args = parser.parse_args()

    if not 0 <= args.self_replace <= 1:
        parser.error("--self-replace must be between 0 and 1.")
    if not all(1 <= age <= 100 for age in [args.age, *args.targets]):
        parser.error("Ages must be between 1 and 100.")

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image not found: {image}")

    repo = Path.home() / ".cache/fading-src"
    model = Path.home() / ".cache/fading-models/finetune_double_prompt_150_random"
    entry = repo / "age_editing.py"
    if not entry.is_file() or not model.is_dir():
        parser.error("Install the official FADING code and checkpoint separately.")

    source = entry.read_text(encoding="utf-8")
    marker = "self_replace_steps = .5"
    if source.count(marker) != 1:
        raise RuntimeError("Official source changed; review it before running.")
    source = source.replace(
        marker,
        f"self_replace_steps = {args.self_replace!r}",
        1,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path.home() / ".cache/fading-results" / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    (output / "settings.json").write_text(
        __import__("json").dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(repo))
    sys.argv = [
        str(entry),
        "--image_path", str(image),
        "--age_init", str(args.age),
        "--gender", args.gender,
        "--specialized_path", str(model),
        "--save_aged_dir", str(output),
        "--target_ages", *map(str, args.targets),
    ]

    print("Results:", output, flush=True)
    started = perf_counter()
    namespace = {"__name__": "__main__", "__file__": str(entry)}
    exec(compile(source, str(entry), "exec"), namespace)
    print(f"Total seconds: {perf_counter() - started:.1f}", flush=True)

if __name__ == "__main__":
    main()
