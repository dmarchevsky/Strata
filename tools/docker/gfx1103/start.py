"""Start Strata's server for Inkling-2-Mini inside the gfx1103 container, from environment variables.

Settings (all but MODEL optional):
  MODEL            the model file inside the container: a .strata (SQ-10 / SQ-15 / SQ-20) or an Unsloth .gguf
  VISION           auto (default: Inkling-2-Mini-Vision.strata beside the model, else none) | off | a path to a Vision .strata
  CONTEXT          tokens: 32k, 64k, 128k, 256k or a number (default 32k)
  PARALLEL         conversations answered at the same time, 1 to 64 (default 1)
  DRAFTING         off (default) | dspark (780M, SQ-15: faster on code and edits, about 6% slower on prose)
  GPU              the HIP device number (default 0: on a PC whose other card ROCm skips, the 780M)
  MODEL_NAME       the name clients see in /v1/models (default: from the file name)
  API_KEY          the key clients must send; empty = no key (the port is published on 127.0.0.1 by default)
  HIPBLASLT_TABLE  on (default) | off: the gfx1103 hipBLASLt table. Only the Qwen engine reads it; strata-inkling does not
                   (it keeps hipBLASLt off on RDNA3: STRATA_HIPBLASLT=1 measured 26% slower prompts here, 252 -> 186 tok/s at 4K)
  TOKENIZER        a tokenizer folder; empty = made by the server in /data/tokenizer/<MODEL_NAME>
  TOKENIZER_SOURCE for a .gguf: the folder with tokenizer.json + chat_template.jinja (default: beside the model or the Vision file)
  ENGINE_ARGS      extra strata-inkling flags, space separated (e.g. "--kv k8v8")
  SERVER_ARGS      extra serve/server.py flags, space separated
Any STRATA_* variable is passed to the server and the engine as it is.
"""
import json
import os
import re
import shlex
import sys
from pathlib import Path

TERMINAL = Path("/opt/strata/terminal")
EXE = "/opt/strata/bin/strata-inkling"
TABLE = TERMINAL / "tools/hip/gfx1103-hipblaslt-100401.txt"
DATA = Path("/data")


def die(msg: str) -> None:
    print(f"[X] {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def tokens(text: str) -> int:
    m = re.fullmatch(r"(\d+)\s*([kK]?)", text)
    if not m:
        die(f"CONTEXT={text!r}: use a number of tokens or 32k / 64k / 128k / 256k")
    return int(m.group(1)) * (1024 if m.group(2) else 1)


model = env("MODEL")
if not model:
    die("MODEL is not set: give the model file as seen inside the container, e.g. "
        "/models/Strata/models/inkling-2-mini/Inkling-2-Mini-SQ-15.strata")
model_path = Path(model)
if not model_path.is_file():
    die(f"MODEL={model}: no such file inside the container (is the models folder mounted at /models?)")
if model_path.suffix.lower() not in (".strata", ".gguf"):
    die(f"MODEL={model}: expected a .strata or .gguf file")

vision = env("VISION", "auto")
if vision.lower() == "auto":
    beside = model_path.parent / "Inkling-2-Mini-Vision.strata"
    vision = str(beside) if beside.is_file() else ""
elif vision.lower() in ("off", "none", "0", "false"):
    vision = ""
elif not Path(vision).is_file():
    die(f"VISION={vision}: no such file inside the container")

context = tokens(env("CONTEXT", "32k"))
parallel = env("PARALLEL", "1")
if not parallel.isdigit() or not 1 <= int(parallel) <= 64:
    die(f"PARALLEL={parallel!r}: use a number from 1 to 64")
drafting = env("DRAFTING", "off").lower()
if drafting not in ("off", "dspark"):
    die(f"DRAFTING={drafting!r}: use off or dspark")
gpu = env("GPU", "0")
if not gpu.isdigit():
    die(f"GPU={gpu!r}: use one device number, e.g. 0")
name = env("MODEL_NAME") or re.sub(r"[^a-z0-9._-]+", "-", model_path.stem.lower())

args = ["--native", str(model_path), "--max-context", str(context)]
if vision:
    args += ["--vision", vision]
args += shlex.split(env("ENGINE_ARGS"))

tokenizer = env("TOKENIZER") or str(DATA / "tokenizer" / name)
for d in (DATA / "logs", DATA / "home", DATA / "state", DATA / "config", Path(tokenizer)):
    d.mkdir(parents=True, exist_ok=True)

# A .strata model carries its tokenizer and the server exports it. A .gguf one does not: Strata's setup downloads
# tokenizer.json + chat_template.jinja for it. Use the pair found beside the model, else beside the Vision file
# (the Strata model folder), else in TOKENIZER_SOURCE; the server turns them into its own files in the tokenizer folder.
if model_path.suffix.lower() == ".gguf" and not (Path(tokenizer) / "vocab.json").exists():
    import shutil
    places = [Path(env("TOKENIZER_SOURCE"))] if env("TOKENIZER_SOURCE") else []
    places += [model_path.parent] + ([Path(vision).parent] if vision else [])
    src = next((p for p in places if (p / "tokenizer.json").is_file() and (p / "chat_template.jinja").is_file()), None)
    if src is None:
        die("this .gguf model has no tokenizer.json + chat_template.jinja beside it or beside the Vision file: "
            "set TOKENIZER_SOURCE to the folder that holds them (Strata's Inkling model folder has both)")
    for f in ("tokenizer.json", "chat_template.jinja"):
        shutil.copyfile(src / f, Path(tokenizer) / f)
    print(f"[strata-780m] tokenizer files from {src}", flush=True)

cfg = {
    "exe": EXE,
    "args": args,
    "cwd": str(TERMINAL),
    "tokenizer": tokenizer,
    "model_name": name,
    "arch": "inkling2_loop",
    "log": str(DATA / "logs" / f"engine-{name}.log"),
    "lib_dirs": [],
    "port": 8080,
    "drafting": drafting,
    "parallel": int(parallel),
    "backend": "hip",
    "gpu": int(gpu),
}
config_path = DATA / "config" / "inkling-run.json"
config_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")

table = env("HIPBLASLT_TABLE", "on").lower()
if table in ("on", "1", "true", "yes"):
    os.environ.setdefault("STRATA_HIPBLASLT_TUNING", str(TABLE))
elif table in ("off", "0", "false", "no"):
    os.environ.pop("STRATA_HIPBLASLT_TUNING", None)
else:
    die(f"HIPBLASLT_TABLE={table!r}: use on or off")

server = [sys.executable, str(TERMINAL / "serve/server.py"), "--engine", "strata", "--config", str(config_path),
          "--host", "0.0.0.0", "--port", "8080"]
key = env("API_KEY") or env("STRATA_API_KEY")
os.environ.pop("STRATA_API_KEY", None)          # an empty STRATA_API_KEY stops the server
os.environ.pop("API_KEY", None)
if key:
    os.environ["STRATA_API_KEY"] = key          # by environment, so the key is not in the process list
else:
    server += ["--no-api-key"]                  # the compose file publishes the port on 127.0.0.1 unless BIND says otherwise
server += shlex.split(env("SERVER_ARGS"))

print(f"[strata-780m] model {model_path.name} as {name!r}, context {context}, parallel {parallel}, drafting {drafting}, "
      f"vision {Path(vision).name if vision else 'off'}, gpu {gpu}, hipBLASLt table {'on' if 'STRATA_HIPBLASLT_TUNING' in os.environ else 'off'}, "
      f"API key {'set' if key else 'none'}", flush=True)
print(f"[strata-780m] engine log: {cfg['log']}", flush=True)
os.chdir(TERMINAL)
os.execv(sys.executable, server)
