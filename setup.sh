#!/bin/bash
# setup.sh - Create isolated child environments for GPU kernel optimization
#
# Usage:
#   bash setup.sh --operator <name>                                 # local, auto-detect GPU, from scratch
#   bash setup.sh --operator <name> --name "experiment_1"           # local with label
#   bash setup.sh --operator <name> --backend modal --gpu b200      # Modal B200, from scratch
#   bash setup.sh --operator <name> --mode existing --kernel /path/to/kernel.py
#   bash setup.sh --operator <name> --dataset /path/to/dataset      # custom dataset
#   bash setup.sh --operator <name> --language cuda                 # CUDA kernel (default: triton)
#   bash setup.sh --operator <name> --task /path/to/task.md         # custom task template
#   bash setup.sh --operator <name> --hints /path/to/hints.md      # custom hints
#   bash setup.sh                                                   # list available operators

set -euo pipefail

# --- Defaults ---
MODE="scratch"
BACKEND="local"
LANGUAGE="triton"
GPU=""
AGENT="claude"
KERNEL_PATH=""
LABEL=""
OPERATOR=""
DATASET_PATH=""
TASK_PATH=""
HINTS_PATH=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --kernel)
            KERNEL_PATH="$2"
            shift 2
            ;;
        --name)
            LABEL="$2"
            shift 2
            ;;
        --operator)
            OPERATOR="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --task|--prompt)
            TASK_PATH="$2"
            shift 2
            ;;
        --hints|--info)
            HINTS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Usage: bash setup.sh --operator <name> [--dataset <path>] [--mode scratch|existing] [--backend local|modal] [--language triton|cuda] [--gpu <name>] [--agent claude] [--kernel <path>] [--name <label>] [--task <path>] [--hints <path>]"
            exit 1
            ;;
    esac
done

# --- Resolve paths ---
PARENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$PARENT_DIR")"

# --- Derive LANGUAGE_NAME and GPU ---
case "$LANGUAGE" in
    triton) LANGUAGE_NAME="Triton" ;;
    cuda)   LANGUAGE_NAME="CUDA" ;;
    *)
        echo "Error: --language must be 'triton' or 'cuda' (got '$LANGUAGE')"
        exit 1
        ;;
esac

# --- Resolve GPU ---
if [[ -z "$GPU" ]]; then
    if [[ "$BACKEND" == "modal" ]]; then
        echo "Error: --gpu is required for modal backend"
        exit 1
    fi
    # Local mode: auto-detect via nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: No GPU specified and nvidia-smi not found. Use --gpu <name>."
        exit 1
    fi
    GPU_FULL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -z "$GPU_FULL" ]]; then
        echo "Error: No GPU detected. Use --gpu <name>."
        exit 1
    fi
    # Extract model identifier (A100, H100, B200, etc.)
    GPU=$(echo "$GPU_FULL" | grep -oP '[A-Z]\d+' | head -1 | tr '[:upper:]' '[:lower:]')
    if [[ -z "$GPU" ]]; then
        echo "Error: Could not identify GPU model from '$GPU_FULL'. Use --gpu <name>."
        exit 1
    fi
    echo "Auto-detected GPU: $GPU_FULL (using --gpu $GPU)"
fi

# GPU_NAME = uppercase of slug (a100 → A100, h100 → H100)
GPU_NAME=$(echo "$GPU" | tr '[:lower:]' '[:upper:]')

# --- Resolve agent config ---
AGENT_CONFIG="$PARENT_DIR/templates/agent/${AGENT}.json"
if [[ ! -f "$AGENT_CONFIG" ]]; then
    echo "Error: Agent config not found: $AGENT_CONFIG"
    exit 1
fi
TASK_FILENAME=$(python3 -c "import json; print(json.load(open('$AGENT_CONFIG'))['task_filename'])")
HINTS_FILENAME=$(python3 -c "import json; print(json.load(open('$AGENT_CONFIG'))['hints_filename'])")

# --- Resolve dataset path ---
if [[ -n "$DATASET_PATH" ]]; then
    FIB_DATASET_PATH="$DATASET_PATH"
else
    FIB_DATASET_PATH="${FIB_DATASET_PATH:-}"
fi
if [[ -z "$FIB_DATASET_PATH" ]]; then
    echo "Error: No dataset path specified."
    echo "Set FIB_DATASET_PATH or use --dataset to specify the path to the flashinfer-bench trace set."
    exit 1
fi
if [[ ! -d "$FIB_DATASET_PATH/definitions" ]]; then
    echo "Error: Dataset not found at $FIB_DATASET_PATH/definitions"
    echo "Set FIB_DATASET_PATH or use --dataset to specify the path to the flashinfer-bench trace set."
    exit 1
fi

# --- Resolve task and hints paths ---
if [[ -z "$TASK_PATH" ]]; then
    TASK_PATH="$PARENT_DIR/templates/task.md"
fi
if [[ ! -f "$TASK_PATH" ]]; then
    echo "Error: Task template not found: $TASK_PATH"
    exit 1
fi

if [[ -z "$HINTS_PATH" ]]; then
    HINTS_PATH="$PARENT_DIR/templates/hints.md"
fi
if [[ ! -f "$HINTS_PATH" ]]; then
    echo "Error: Hints file not found: $HINTS_PATH"
    exit 1
fi

# --- List operators if none specified ---
if [[ -z "$OPERATOR" ]]; then
    echo "Available operators:"
    echo ""
    for def_file in "$FIB_DATASET_PATH"/definitions/*/*.json; do
        op_name="$(basename "$def_file" .json)"
        op_type="$(basename "$(dirname "$def_file")")"
        echo "  $op_name  ($op_type)"
    done
    echo ""
    echo "Usage: bash setup.sh --operator <name> [--dataset <path>] [--mode scratch|existing] [--backend local|modal] [--language triton|cuda] [--gpu <name>] [--agent claude] [--kernel <path>] [--name <label>] [--task <path>] [--hints <path>]"
    exit 0
fi

# --- Auto-discover operator definition ---
DEFINITION_FILE=""
for def_file in "$FIB_DATASET_PATH"/definitions/*/"${OPERATOR}.json"; do
    if [[ -f "$def_file" ]]; then
        DEFINITION_FILE="$def_file"
        break
    fi
done

if [[ -z "$DEFINITION_FILE" ]]; then
    echo "Error: Operator '$OPERATOR' not found in $FIB_DATASET_PATH/definitions/"
    echo "Run without --operator to list available operators."
    exit 1
fi

# Find workloads file
OP_TYPE="$(basename "$(dirname "$DEFINITION_FILE")")"
WORKLOADS_FILE="$FIB_DATASET_PATH/workloads/${OP_TYPE}/${OPERATOR}.jsonl"
if [[ ! -f "$WORKLOADS_FILE" ]]; then
    echo "Error: Workloads file not found: $WORKLOADS_FILE"
    exit 1
fi

echo "Operator: $OPERATOR ($OP_TYPE)"
echo "Definition: $DEFINITION_FILE"
echo "Workloads: $WORKLOADS_FILE"
echo "Dataset: $FIB_DATASET_PATH"

# --- Validate ---
if [[ "$MODE" != "scratch" && "$MODE" != "existing" ]]; then
    echo "Error: --mode must be 'scratch' or 'existing' (got '$MODE')"
    exit 1
fi

if [[ "$BACKEND" != "local" && "$BACKEND" != "modal" ]]; then
    echo "Error: --backend must be 'local' or 'modal' (got '$BACKEND')"
    exit 1
fi

if [[ "$MODE" == "existing" ]]; then
    if [[ -z "$KERNEL_PATH" ]]; then
        echo "Error: --mode existing requires --kernel <path>"
        exit 1
    fi
    if [[ ! -f "$KERNEL_PATH" ]]; then
        echo "Error: Kernel file not found: $KERNEL_PATH"
        exit 1
    fi
    if ! grep -q 'def run(' "$KERNEL_PATH"; then
        echo "Error: Kernel file does not contain 'def run(': $KERNEL_PATH"
        exit 1
    fi
fi

# --- Validate fragment files exist ---
GPU_FRAGMENT="$PARENT_DIR/templates/fragments/gpu-${GPU}.md"
BACKEND_FRAGMENT="$PARENT_DIR/templates/fragments/backend-${BACKEND}.md"
OBJECTIVE_FRAGMENT="$PARENT_DIR/templates/fragments/objective-${MODE}.md"

# Backend and objective fragments are required; GPU fragment is optional
for frag in "$BACKEND_FRAGMENT" "$OBJECTIVE_FRAGMENT"; do
    if [[ ! -f "$frag" ]]; then
        echo "Error: Fragment not found: $frag"
        exit 1
    fi
done

# --- Generate template context ---
CONTEXT_FILE=$(mktemp)
trap 'rm -f "$CONTEXT_FILE"' EXIT

python3 "$PARENT_DIR/scripts/generate_context.py" \
    --definition "$DEFINITION_FILE" \
    --workloads "$WORKLOADS_FILE" \
    --output "$CONTEXT_FILE"

# shellcheck disable=SC1090
source "$CONTEXT_FILE"

# --- Auto-increment run number ---
MAX_NUM=0
shopt -s nullglob
for dir in "$BASE_DIR"/kernel-opt-agent-run-*/; do
    basename="$(basename "$dir")"
    # Extract the 3-digit number after "kernel-opt-agent-run-"
    num_part="${basename#kernel-opt-agent-run-}"
    num="${num_part%%-*}"  # strip everything after first dash (label)
    # Check if it's a valid number
    if [[ "$num" =~ ^[0-9]+$ ]]; then
        num=$((10#$num))  # force base-10
        if (( num > MAX_NUM )); then
            MAX_NUM=$num
        fi
    fi
done
shopt -u nullglob

NEXT_NUM=$(printf "%03d" $((MAX_NUM + 1)))

if [[ -n "$LABEL" ]]; then
    CHILD_NAME="kernel-opt-agent-run-${NEXT_NUM}-${LABEL}"
else
    CHILD_NAME="kernel-opt-agent-run-${NEXT_NUM}"
fi

CHILD_DIR="$BASE_DIR/$CHILD_NAME"

# --- Create child directory structure ---
mkdir -p "$CHILD_DIR/.claude"
mkdir -p "$CHILD_DIR/docs"
mkdir -p "$CHILD_DIR/scripts"
mkdir -p "$CHILD_DIR/solution/${LANGUAGE}"

# --- Copy common files ---
cp "$PARENT_DIR/.gitignore" "$CHILD_DIR/.gitignore"
cp "$DEFINITION_FILE" "$CHILD_DIR/docs/definition.json"
cp "$WORKLOADS_FILE" "$CHILD_DIR/docs/workloads.jsonl"
cp "$PARENT_DIR/scripts/pack_solution.py" "$CHILD_DIR/scripts/pack_solution.py"
cp "$PARENT_DIR/scripts/bench_utils.py" "$CHILD_DIR/scripts/bench_utils.py"

# --- Copy hints file ---
cp "$HINTS_PATH" "$CHILD_DIR/$HINTS_FILENAME"

# --- Generate config.toml ---
cat > "$CHILD_DIR/config.toml" <<TOML
[solution]
name = "${OPERATOR}-reference"
definition = "${OPERATOR}"
author = "user"

[build]
language = "${LANGUAGE}"
gpu = "${GPU}"
entry_point = "kernel.py::run"
destination_passing_style = false
TOML

# --- Copy backend-specific scripts ---
if [[ "$BACKEND" == "local" ]]; then
    cp "$PARENT_DIR/scripts/bench.sh" "$CHILD_DIR/scripts/bench.sh"
    cp "$PARENT_DIR/scripts/run_local.py" "$CHILD_DIR/scripts/run_local.py"
else
    cp "$PARENT_DIR/scripts/bench_modal.sh" "$CHILD_DIR/scripts/bench.sh"
    cp "$PARENT_DIR/scripts/run_modal.py" "$CHILD_DIR/scripts/run_modal.py"
fi

# --- Generate .claude/settings.local.json from agent config ---
python3 -c "
import json
with open('$AGENT_CONFIG') as f:
    config = json.load(f)
settings = {'permissions': config['permissions']}
with open('$CHILD_DIR/.claude/settings.local.json', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
"

# --- Copy kernel (and config.toml if colocated with kernel) ---
if [[ "$MODE" == "scratch" ]]; then
    # Extract reference kernel from definition.json
    python3 -c "
import json, sys
with open('$DEFINITION_FILE') as f:
    d = json.load(f)
ref = d.get('reference', '')
if not ref:
    print('Error: No reference field in definition.json', file=sys.stderr)
    sys.exit(1)
with open('$CHILD_DIR/solution/${LANGUAGE}/kernel.py', 'w') as f:
    f.write(ref)
"
else
    cp "$KERNEL_PATH" "$CHILD_DIR/solution/${LANGUAGE}/kernel.py"
    # If a config.toml exists next to the kernel (e.g. from a trajectory snapshot),
    # use it to preserve destination_passing_style and other settings
    KERNEL_DIR="$(dirname "$KERNEL_PATH")"
    if [[ -f "$KERNEL_DIR/config.toml" ]]; then
        cp "$KERNEL_DIR/config.toml" "$CHILD_DIR/config.toml"
        echo "Note: Using config.toml from $(realpath "$KERNEL_DIR/config.toml")"
    fi
fi

# --- Read fragment contents ---
if [[ -f "$GPU_FRAGMENT" ]]; then
    GPU_CONTENT=$(cat "$GPU_FRAGMENT")
else
    GPU_CONTENT="- GPU: $GPU_NAME"
fi
BACKEND_CONTENT=$(cat "$BACKEND_FRAGMENT")
OBJECTIVE_RAW=$(cat "$OBJECTIVE_FRAGMENT")

# --- Generate task file (CLAUDE.md) ---
# First pass: replace variables in objective fragment
OBJECTIVE="${OBJECTIVE_RAW//\{\{LANGUAGE_NAME\}\}/$LANGUAGE_NAME}"
OBJECTIVE="${OBJECTIVE//\{\{LANGUAGE\}\}/$LANGUAGE}"
OBJECTIVE="${OBJECTIVE//\{\{GPU_NAME\}\}/$GPU_NAME}"
OBJECTIVE="${OBJECTIVE//\{\{OPERATOR_DESCRIPTION\}\}/$OPERATOR_DESCRIPTION}"
OBJECTIVE="${OBJECTIVE//\{\{OPERATOR\}\}/$OPERATOR}"

# Second pass: assemble task.md with all placeholders
awk \
    -v objective="$OBJECTIVE" \
    -v gpu="$GPU_CONTENT" \
    -v backend="$BACKEND_CONTENT" \
    -v operator="$OPERATOR" \
    -v operator_desc="$OPERATOR_DESCRIPTION" \
    -v num_workloads="$NUM_WORKLOADS" \
    -v shape_summary="$SHAPE_SUMMARY" \
    -v workload_summary="$WORKLOAD_SUMMARY" \
    -v language="$LANGUAGE" \
    -v language_name="$LANGUAGE_NAME" \
    -v gpu_name="$GPU_NAME" \
    '
    /\{\{OBJECTIVE\}\}/ { print objective; next }
    /\{\{GPU\}\}/ { print gpu; next }
    /\{\{BACKEND\}\}/ { print backend; next }
    /\{\{SHAPE_SUMMARY\}\}/ { print shape_summary; next }
    /\{\{WORKLOAD_SUMMARY\}\}/ { print workload_summary; next }
    {
        gsub(/\{\{OPERATOR\}\}/, operator)
        gsub(/\{\{OPERATOR_DESCRIPTION\}\}/, operator_desc)
        gsub(/\{\{NUM_WORKLOADS\}\}/, num_workloads)
        gsub(/\{\{LANGUAGE\}\}/, language)
        gsub(/\{\{LANGUAGE_NAME\}\}/, language_name)
        gsub(/\{\{GPU_NAME\}\}/, gpu_name)
        print
    }
' "$TASK_PATH" > "$CHILD_DIR/$TASK_FILENAME"

# --- Git init ---
cd "$CHILD_DIR"
git init -q
git add -A
git commit -q -m "Initial commit (spawned from kernel-opt-agent, operator=$OPERATOR, mode=$MODE, backend=$BACKEND, language=$LANGUAGE)"

# --- Summary ---
echo ""
echo "===== Child environment created ====="
echo "  Path:     $CHILD_DIR"
echo "  Operator: $OPERATOR"
echo "  Mode:     $MODE"
echo "  Backend:  $BACKEND"
echo "  Language: $LANGUAGE"
echo "  GPU:      $GPU_NAME"
echo "  Dataset:  $FIB_DATASET_PATH"
if [[ "$MODE" == "existing" ]]; then
    echo "  Kernel:   $KERNEL_PATH"
fi
echo ""
echo "Next steps:"
echo "  cd $CHILD_DIR"
echo ""
echo "  # Start your code agent, e.g.:"
echo "  claude"
echo ""
echo "To run benchmark manually:"
echo "  bash scripts/bench.sh"
echo "======================================="
