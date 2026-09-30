#!/usr/bin/env bash
# Project container on the Jetson. No sudo needed (account is in the docker group).
#
#   jetson/run.sh build            build the image (once, ~5-10 min, needs internet)
#   jetson/run.sh shell            interactive shell, repo bind-mounted at /workspace
#   jetson/run.sh run <cmd...>     one-off command in a fresh container, e.g.
#                                  jetson/run.sh run python3 task1_rtmpose/03_infer_frames.py --help
#   jetson/run.sh up               start a persistent detached container named $CONTAINER
#   jetson/run.sh exec <cmd...>    run a command inside the persistent container
#   jetson/run.sh down             stop + remove the persistent container
#   jetson/run.sh probe            print the stack versions + GPU + power-rail sanity inside the container
#   jetson/run.sh whoelse         is the shared board free? other users, load, GPU/CPU hogs, free MemFree.
#                                 Run this BEFORE any timing or power measurement -- another tenant
#                                 inflates latency and power, and nothing in the result would show it.
#   jetson/run.sh clean-large      delete weights / ONNX / engines / frames from the board (shared scratch)
#
# Env overrides: SLT_PROJECT (host repo path, default ~/sign-lang-project),
#                SLT_IMAGE (default slt-jetson:25.06), SLT_CONTAINER (default slt-work).
set -euo pipefail

PROJ="${SLT_PROJECT:-$HOME/sign-lang-project}"
IMAGE="${SLT_IMAGE:-slt-jetson:25.06}"
CONTAINER="${SLT_CONTAINER:-slt-work}"
CACHE="$HOME/.cache/slt-container"          # HF / torch / pip caches survive container restarts
UNAME="$(id -un)"

mkdir -p "$CACHE"

# Flags shared by every container: GPU runtime, host network (no port mapping fuss), the shmem /
# ulimit settings the NGC image asks for, the repo at /workspace, caches persisted, run as you.
COMMON_FLAGS=(
    --runtime nvidia
    --network host
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --user "$(id -u):$(id -g)"
    -v "$PROJ:/workspace"
    -v "$CACHE:/home/$UNAME/.cache"
    -v /var/lib/nvpmodel/status:/var/lib/nvpmodel/status:ro   # power mode visible to envinfo.py
    -w /workspace
)

cmd="${1:-}"; shift || true
case "$cmd" in
    build)
        docker build \
            --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg UNAME="$UNAME" \
            -t "$IMAGE" -f "$PROJ/jetson/Dockerfile" "$PROJ/jetson"
        ;;
    shell)
        docker run --rm -it "${COMMON_FLAGS[@]}" "$IMAGE" bash
        ;;
    run)
        docker run --rm "${COMMON_FLAGS[@]}" "$IMAGE" "$@"
        ;;
    up)
        if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            docker start "$CONTAINER" >/dev/null
        else
            docker run -d --name "$CONTAINER" "${COMMON_FLAGS[@]}" "$IMAGE" sleep infinity >/dev/null
        fi
        echo "$CONTAINER is up: jetson/run.sh exec <cmd>"
        ;;
    exec)
        docker exec -it -w /workspace "$CONTAINER" "$@"
        ;;
    exec-batch)   # non-tty variant for scripted / ssh use
        docker exec -w /workspace "$CONTAINER" "$@"
        ;;
    down)
        docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "removed $CONTAINER" || echo "$CONTAINER not running"
        ;;
    clean-large)   # the board is shared scratch: pull results to the Mac first, then run this
        cd "$PROJ"
        echo "before: $(du -sh . | cut -f1)"
        rm -rf weights models/*.onnx models/*.engine models/*/*.onnx models/*/*.engine \
               models/mt5_pruned_onnx data/test_frames data/calib_frames data/clips \
               data/openasl_test_pose data/openasl_train_pose results/*.npz
        echo "after:  $(du -sh . | cut -f1)   (results/*.json|csv|md and the code are kept)"
        ;;
    whoelse)   # occupancy check: timing/power runs are only valid on an idle board
        echo "== logged-in users (want: only you, or none)"; who
        echo "== load average (want: < ~0.5 before you start)"; uptime
        echo "== processes over 1% CPU not owned by you"
        ps -eo user,pcpu,pmem,etime,args --sort=-pcpu \
            | awk -v me="$(id -un)" 'NR==1 || ($1!=me && $2+0>1.0)' | head -15
        echo "== containers running (other tenants work in theirs)"
        docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
        echo "== GPU clients (any process holding /dev/nvidia*)"
        fuser -v /dev/nvidia* 2>&1 | head -10 || echo "  (fuser unavailable)"
        echo "== memory (nvmap allocates from MemFree only; see jetson/drop_file_cache.py)"
        free -m | head -2
        echo "== power mode (0 = 15 W; every published row is mode 0)"
        cat /var/lib/nvpmodel/status 2>/dev/null || echo "  status file not visible"
        ;;
    probe)
        docker run --rm "${COMMON_FLAGS[@]}" "$IMAGE" python3 - <<'PY'
import sys, os, glob
print("python", sys.version.split()[0], "user", os.environ.get("USER"), "home", os.environ.get("HOME"))
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import tensorrt as trt; print("tensorrt", trt.__version__)
for m in ("numpy", "cv2", "onnx", "onnxruntime", "transformers", "sentencepiece"):
    try: print(m, __import__(m).__version__)
    except Exception as e: print(m, "MISSING", e)
print("nvpmodel:", open("/var/lib/nvpmodel/status").read().strip() if os.path.exists("/var/lib/nvpmodel/status") else "status file not visible")
rails = sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"))
for lab in rails:
    n = os.path.basename(lab)[2:-6]
    d = os.path.dirname(lab)
    try:
        v = int(open(f"{d}/in{n}_input").read()); i = int(open(f"{d}/curr{n}_input").read())
        print(f"rail {open(lab).read().strip():16s} {v} mV  {i} mA  {v*i/1e6:.2f} W")
    except Exception as e:
        print("rail", open(lab).read().strip(), "unreadable:", e)
PY
        ;;
    *)
        sed -n '2,15p' "$0"; exit 1
        ;;
esac
