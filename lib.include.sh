#!/usr/bin/env bash

set -e

export SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
readonly SCRIPT_DIR
cd -- "${SCRIPT_DIR}"

# User-configurable environment variables (pass via environment, don't modify below).
# Conda detection: user value > $CONDA_EXE > "conda" from PATH.
# This order is required because Conda's init script shadows "conda" as a shell function.
export OT_CONDA_CMD="${OT_CONDA_CMD:-${CONDA_EXE:-conda}}"
export OT_CONDA_ENV="${OT_CONDA_ENV:-conda_env}"
export OT_PYTHON_CMD="${OT_PYTHON_CMD:-python}"
export OT_PYTHON_VENV="${OT_PYTHON_VENV:-venv}"
export OT_PREFER_VENV="${OT_PREFER_VENV:-false}"
export OT_LAZY_UPDATES="${OT_LAZY_UPDATES:-false}"
export OT_CUDA_LOWMEM_MODE="${OT_CUDA_LOWMEM_MODE:-false}"
export OT_PLATFORM_REQUIREMENTS="${OT_PLATFORM_REQUIREMENTS:-detect}"
export OT_SCRIPT_DEBUG="${OT_SCRIPT_DEBUG:-false}"

export OT_PYTHON_VERSION_MINIMUM="3.10"
export OT_PYTHON_VERSION_TOO_HIGH="3.14"
export OT_CONDA_USE_PYTHON_VERSION="3.13"
export OT_MUST_INSTALL_REQUIREMENTS="false"
export OT_UPDATE_METADATA_FILE="${SCRIPT_DIR}/update.var"
export OT_HOST_OS="$(uname -s)"

if [[ "${OT_HOST_OS}" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK="1"
fi

if [[ "${OT_CUDA_LOWMEM_MODE}" == "true" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
fi

function escape_shell_command {
    printf " %q" "$@" | sed 's/^ //'
}

function print {
    printf "[OneTrainer] %b\n" "$*"
}

function print_warning {
    printf "[OneTrainer] Warning: %b\n" "$*" >&2
}

function print_error {
    printf "[OneTrainer] Error: %b\n" "$*" >&2
}

function print_debug {
    if [[ "${OT_SCRIPT_DEBUG}" == "true" ]]; then
        print "Debug: $*"
    fi
}

function print_command {
    printf "[OneTrainer] + %s\n" "$(escape_shell_command "$@")"
}

function get_current_git_hash {
    git rev-parse HEAD
}

function save_update_metadata {
    get_current_git_hash >"${OT_UPDATE_METADATA_FILE}"
}

function is_update_metadata_changed {
    if [[ -f "${OT_UPDATE_METADATA_FILE}" ]]; then
        local saved_hash="$(<"${OT_UPDATE_METADATA_FILE}")"
        local current_hash="$(get_current_git_hash)"
        print_debug "Saved Metadata Hash=\"${saved_hash}\", Current Hash=\"${current_hash}\""
        if [[ "${saved_hash}" == "${current_hash}" ]]; then
            return 1
        fi
    fi

    return 0
}

function regex_escape {
    sed 's/[][\.|$(){}?+*^]/\\&/g' <<<"$*"
}

function absolute_path {
    if [[ -z "$1" ]]; then
        print_error "absolute_path requires 1 argument."
        return 1
    fi

    if [[ ! -d "$1" ]]; then
        print_error "absolute_path argument is not a directory: \"$1\"."
        return 1
    fi

    echo "$(cd -- "$1" &>/dev/null && pwd)"
}

function can_exec {
    if [[ -z "$1" ]]; then
        print_error "can_exec requires 1 argument."
        return 1
    fi

    if local full_path="$(command -v "$1" 2>/dev/null)"; then
        if [[ ! -z "${full_path}" ]] && [[ -x "${full_path}" ]]; then
            return 0
        fi
    fi

    return 1
}

function run_cmd {
    print_command "$@"
    "$@"
}

function run_python {
    run_cmd "${OT_PYTHON_CMD}" "$@"
}

function run_pip {
    run_python -m pip "$@"
}

function run_venv {
    run_python -m venv "$@"
}

function has_python {
    can_exec "${OT_PYTHON_CMD}"
}

function has_python_venv {
    [[ -f "${OT_PYTHON_VENV}/bin/activate" ]]
}

function create_python_venv {
    print "Creating Python Venv environment in \"${OT_PYTHON_VENV}\"..."
    run_venv "${OT_PYTHON_VENV}"
    export OT_MUST_INSTALL_REQUIREMENTS="true"
}

function ensure_python_venv_exists {
    if ! has_python_venv; then
        create_python_venv
    fi
}

function activate_python_venv {
    source "${OT_PYTHON_VENV}/bin/activate"

    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_error "Something went wrong when activating the Python Venv in \"${OT_PYTHON_VENV}\"."
        exit 1
    fi

    # Venv's Python binary is always named "python" regardless of OT_PYTHON_CMD.
    export OT_PYTHON_CMD="python"
}

function run_conda {
    run_cmd "${OT_CONDA_CMD}" "$@"
}

__HAS_CONDA__CACHE=""
function has_conda {
    if [[ -z "${__HAS_CONDA__CACHE}" ]]; then
        if can_exec "${OT_CONDA_CMD}"; then
            __HAS_CONDA__CACHE="true"
        else
            __HAS_CONDA__CACHE="false"
        fi
    fi

    [[ "${__HAS_CONDA__CACHE}" == "true" ]]
}

function has_conda_env {
    [[ -d "${OT_CONDA_ENV}/conda-meta" ]]
}

function has_conda_global_env {
    if [[ -z "$1" ]]; then
        print_error "has_conda_global_env requires 1 argument."
        return 1
    fi

    run_conda info --envs | grep -q -- "^$(regex_escape "$1")\b"
}

function create_conda_env {
    print "Creating Conda environment in \"${OT_CONDA_ENV}\"..."

    # ".*" suffix gets latest patch release for the specified Python version.
    declare -a install_args=()
    install_args+=("python==${OT_CONDA_USE_PYTHON_VERSION}.*")

    # Linux needs conda-forge's xft Tk variant for working fonts/Unicode/antialiasing.
    if [[ "${OT_HOST_OS}" == "Linux" ]]; then
        install_args+=("tk[build=xft_*]")
    fi

    # Strict channel priority prevents mixing packages from conda-forge and defaults.
    run_conda create -y --prefix "${OT_CONDA_ENV}" --override-channels --strict-channel-priority --channel "conda-forge" "${install_args[@]}"
    export OT_MUST_INSTALL_REQUIREMENTS="true"

    if has_conda_global_env "ot"; then
        print_warning "The deprecated \"ot\" Conda environment has been detected on your system. It is occupying several gigabytes of disk space, and can be deleted manually to reclaim the storage space.\n\nTo delete the outdated Conda environment, execute the following command:\n\"${OT_CONDA_CMD}\" remove -y --name \"ot\" --all"
    fi
}

function ensure_conda_env_exists {
    if ! has_conda_env; then
        create_conda_env
    fi
}

function run_in_conda_env {
    run_conda run --prefix "${OT_CONDA_ENV}" --no-capture-output "$@"
}

function run_python_in_conda_env {
    run_in_conda_env python "$@"
}

function run_pip_in_conda_env {
    run_python_in_conda_env -m pip "$@"
}

function should_use_conda {
    [[ "${OT_PREFER_VENV}" != "true" ]] && has_conda
}

function activate_chosen_env {
    if should_use_conda; then
        print "Using Conda environment in \"${OT_CONDA_ENV}\"..."
        ensure_conda_env_exists
    else
        print "Using Python Venv environment in \"${OT_PYTHON_VENV}\"..."
        ensure_python_venv_exists
        activate_python_venv
    fi
}

function run_python_in_active_env {
    if should_use_conda; then
        run_python_in_conda_env "$@"
    else
        run_python "$@"
    fi
}

function run_pip_in_active_env {
    if should_use_conda; then
        run_pip_in_conda_env "$@"
    else
        run_pip "$@"
    fi
}

function get_platform_requirements_path {
    local platform_reqs="${OT_PLATFORM_REQUIREMENTS}"
    if [[ "${platform_reqs}" == "detect" ]]; then
        # Prioritize NVIDIA: dual-GPU systems typically have integrated AMD + dedicated NVIDIA.
        if [[ -e "/dev/nvidia0" ]] || can_exec nvidia-smi || can_exec "/usr/lib/wsl/lib/nvidia-smi"; then
            platform_reqs="requirements-cuda.txt"
        elif [[ -e "/dev/kfd" ]]; then
            platform_reqs="requirements-rocm.txt"
        else
            platform_reqs="requirements-default.txt"
        fi
    fi

    if [[ -z "${platform_reqs}" ]] || [[ ! -f "${platform_reqs}" ]]; then
        print_error "Requirements file \"${platform_reqs}\" does not exist."
        return 1
    fi

    echo "${platform_reqs}"
}

function install_requirements_in_active_env {
    # "eager" upgrade strategy ensures existing envs match a fresh install.
    print "Installing requirements in active environment..."
    run_pip_in_active_env install --upgrade --upgrade-strategy eager pip setuptools==81.0.0
    run_pip_in_active_env install --upgrade --upgrade-strategy eager -r requirements-global.txt -r "$(get_platform_requirements_path)"
    export OT_MUST_INSTALL_REQUIREMENTS="false"

    if [[ "${OT_LAZY_UPDATES}" == "true" ]]; then
        print_debug "Saving current update-check metadata to disk..."
        save_update_metadata
    elif [[ -f "${OT_UPDATE_METADATA_FILE}" ]]; then
        print_debug "Deleting outdated update-check metadata from disk..."
        rm -f "${OT_UPDATE_METADATA_FILE}"
    fi
}

function install_requirements_in_active_env_if_necessary {
    if [[ "${OT_MUST_INSTALL_REQUIREMENTS}" != "false" ]]; then
        install_requirements_in_active_env
    fi
}

function show_runtime_solutions {
    if should_use_conda; then
        local conda_env_path="${OT_CONDA_ENV}"
        if has_conda_env; then
            conda_env_path="$(absolute_path "${conda_env_path}")"
        fi

        print "Solution: Switch your Conda environment to the required Python version by deleting your old environment, and then run OneTrainer again.\n\nTo delete the outdated Conda environment, execute the following command:\n\"${OT_CONDA_CMD}\" remove -y --prefix \"${conda_env_path}\" --all"
    else
        print "Solution: Either install the required Python version via pyenv (https://github.com/pyenv/pyenv) and set the project directory's Python version with \"pyenv install <version>\" followed by \"pyenv local <version>\", or install Miniconda if you prefer that we automatically manage everything for you (https://docs.anaconda.com/miniconda/). Remember to manually delete any previous Venv or Conda environment which was created with a different Python version. Read \"LAUNCH-SCRIPTS.md\" for more detailed instructions."
    fi
}

function exit_if_no_runtime {
    if ! should_use_conda && ! has_python; then
        print_error "Python command \"${OT_PYTHON_CMD}\" does not exist on your system."
        show_runtime_solutions
        exit 1
    fi
}

function exit_if_active_env_wrong_python_version {
    if ! run_python_in_active_env "scripts/util/version_check.py" "${OT_PYTHON_VERSION_MINIMUM}" "${OT_PYTHON_VERSION_TOO_HIGH}"; then
        show_runtime_solutions
        exit 1
    fi
}

function prepare_runtime_environment {
    exit_if_no_runtime
    activate_chosen_env
    exit_if_active_env_wrong_python_version

    local force_update="false"
    if [[ "$1" == "upgrade" ]]; then
        if [[ "${OT_LAZY_UPDATES}" == "false" ]] || is_update_metadata_changed; then
            force_update="true"
        fi
    fi
    if [[ "${force_update}" == "true" ]]; then
        print_debug "Triggering a forced update of the environment's dependencies..."
        install_requirements_in_active_env
    else
        install_requirements_in_active_env_if_necessary
    fi

    print "Generating UI schema..."
    if ! run_python_in_active_env -m web.scripts.generate_ui_schema; then
        print_warning "UI schema generation failed. Using existing schema."
    fi

    build_web_ui_if_available
}

function build_web_ui_if_available {
    if ! command -v node &> /dev/null; then
        print "Node.js not found. Skipping web UI build."
        print "To use the web UI, install Node.js from https://nodejs.org/"
        return 0
    fi

    local gui_dir="${SCRIPT_DIR}/web/gui"
    local dist_marker="${gui_dir}/dist/main/main/index.cjs"

    if [[ -f "${dist_marker}" ]] || [[ "${1:-}" == "install" ]]; then
        print "Building web UI..."
        (
            cd "${gui_dir}"
            npm install || { print_warning "npm install failed. Web UI may be stale."; return 0; }
            npm run build:electron || { print_warning "Web UI build failed. Web UI may be stale."; return 0; }
            print "Web UI built successfully."
        )
    else
        print "Web UI not previously built. Run start-web-ui.sh to build and launch."
    fi
}
