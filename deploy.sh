#!/usr/bin/env bash
set -euo pipefail

SERVER_IP="${SERVER_IP:-129.211.217.58}"
SERVER_USER="${SERVER_USER:-root}"
REMOTE_PATH="${REMOTE_PATH:-/home/ubuntu/baoshu_ai}"
ENV_NAME="${ENV_NAME:-agent}"
REMOTE_CONDA_SH="${REMOTE_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
LOCAL_CONDA_SH="${LOCAL_CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
DEPLOY_BUNDLE_NAME="${DEPLOY_BUNDLE_NAME:-.codex_deploy.bundle}"
REMOTE_LOG_FILE="${REMOTE_LOG_FILE:-output.log}"

TEST_MODE="${TEST_MODE:-quick}"
INSTALL_DEPS="${INSTALL_DEPS:-n}"
RUN_REMOTE_TESTS="${RUN_REMOTE_TESTS:-y}"
GIT_RELEASE_ENABLED="${GIT_RELEASE_ENABLED:-y}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
SYNC_MODE="${SYNC_MODE:-safe}"
USE_SYSTEMD="${USE_SYSTEMD:-auto}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-}"
SYSTEMD_CANDIDATES_CSV="${SYSTEMD_CANDIDATES_CSV:-baoshu-ai.service,baoshu_ai.service,baoshu-ai,baoshu_ai,baoshu}"
SYSTEMD_UNIT_SOURCE_PATH="${SYSTEMD_UNIT_SOURCE_PATH:-deploy/systemd/baoshu-ai.service}"
REMOTE_HEALTHCHECK_URL="${REMOTE_HEALTHCHECK_URL:-http://127.0.0.1:8000/api/chat-config}"
HEALTHCHECK_TIMEOUT="${HEALTHCHECK_TIMEOUT:-90}"
HEALTHCHECK_INTERVAL="${HEALTHCHECK_INTERVAL:-3}"
SKIP_HEALTHCHECK="${SKIP_HEALTHCHECK:-n}"
REQUIRED_ENV_VARS="${REQUIRED_ENV_VARS:-DATABASE_URL}"
RECOMMENDED_ENV_VARS="${RECOMMENDED_ENV_VARS:-WECOM_CORPID,WECOM_SECRET,WECOM_TOKEN,WECOM_AES_KEY,WECOM_KF_ID}"
REQUIRE_LLM_PROVIDER="${REQUIRE_LLM_PROVIDER:-y}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.tencentyun.com/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.tencentyun.com}"

LOCAL_BRANCH="$(git branch --show-current || true)"
LOCAL_HEAD="$(git rev-parse HEAD)"
DEPLOY_BUNDLE_REF="${LOCAL_BRANCH:-$LOCAL_HEAD}"

QUICK_TESTS=(
  tests/test_profile_state.py
  tests/test_nodes_unit.py
  tests/test_extractor_eval.py
  tests/test_classifier_eval.py
  tests/test_router_eval.py
  tests/test_execution_eval.py
  tests/test_eval_entrypoints.py
  tests/test_intent_guardrails.py
)
FULL_TESTS=(tests)

usage() {
  cat <<'EOF'
Usage: bash deploy.sh [options]

Options:
  --test-mode <skip|quick|full>   Local test mode. Default: quick
  --skip-tests                    Alias for --test-mode skip
  --install-deps                  Install/update dependencies on server
  --no-install-deps               Skip dependency install on server
  --remote-tests                  Run quick regression tests on server
  --no-remote-tests               Skip remote regression tests
  --no-git-release                Skip local git push/fast-forward release
  --git-remote <name>             Git remote used for release. Default: origin
  --main-branch <name>            Main release branch. Default: main
  --sync-mode <safe|mirror>       safe: no delete, mirror: enable delete
  --allow-delete                  Alias for --sync-mode mirror
  --use-systemd <auto|always|never>
                                  Prefer systemd restart when available
  --service-name <name>           Explicit systemd service name
  --health-url <url>              Remote HTTP health check endpoint
  --health-timeout <seconds>      Health check timeout. Default: 90
  --health-interval <seconds>     Health check interval. Default: 3
  --skip-health-check             Skip HTTP and log health verification
  --required-env <csv>            Required env vars checked on server .env
  --recommended-env <csv>         Recommended env vars warned on missing
  --skip-llm-env-check            Do not require at least one LLM provider config
  -h, --help                      Show this help

Environment overrides:
  SERVER_IP, SERVER_USER, REMOTE_PATH, ENV_NAME, REMOTE_CONDA_SH, LOCAL_CONDA_SH,
  REMOTE_LOG_FILE, GIT_REMOTE, MAIN_BRANCH, SYSTEMD_CANDIDATES_CSV,
  SYSTEMD_UNIT_SOURCE_PATH,
  PIP_INDEX_URL, PIP_TRUSTED_HOST.
EOF
}

log() {
  printf '%s\n' "$1"
}

die() {
  printf '❌ %s\n' "$1" >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

join_csv() {
  local IFS=,
  printf '%s' "$*"
}

validate_yes_no() {
  local label="$1"
  local value="$2"
  [[ "$value" == "y" || "$value" == "n" ]] || die "$label 只能是 y 或 n，当前值: $value"
}

run_cmd() {
  "$@"
}

validate_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label 必须是正整数，当前值: $value"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --test-mode)
        TEST_MODE="${2:?缺少 test mode}"
        shift 2
        ;;
      --skip-tests)
        TEST_MODE="skip"
        shift
        ;;
      --install-deps)
        INSTALL_DEPS="y"
        shift
        ;;
      --no-install-deps)
        INSTALL_DEPS="n"
        shift
        ;;
      --remote-tests)
        RUN_REMOTE_TESTS="y"
        shift
        ;;
      --no-remote-tests)
        RUN_REMOTE_TESTS="n"
        shift
        ;;
      --no-git-release)
        GIT_RELEASE_ENABLED="n"
        shift
        ;;
      --git-remote)
        GIT_REMOTE="${2:?缺少 git remote}"
        shift 2
        ;;
      --main-branch)
        MAIN_BRANCH="${2:?缺少 main branch}"
        shift 2
        ;;
      --sync-mode)
        SYNC_MODE="${2:?缺少 sync mode}"
        shift 2
        ;;
      --allow-delete)
        SYNC_MODE="mirror"
        shift
        ;;
      --use-systemd)
        USE_SYSTEMD="${2:?缺少 use-systemd 参数}"
        shift 2
        ;;
      --service-name)
        SYSTEMD_SERVICE="${2:?缺少 service name}"
        shift 2
        ;;
      --health-url)
        REMOTE_HEALTHCHECK_URL="${2:?缺少 health url}"
        shift 2
        ;;
      --health-timeout)
        HEALTHCHECK_TIMEOUT="${2:?缺少 health timeout}"
        shift 2
        ;;
      --health-interval)
        HEALTHCHECK_INTERVAL="${2:?缺少 health interval}"
        shift 2
        ;;
      --skip-health-check)
        SKIP_HEALTHCHECK="y"
        shift
        ;;
      --required-env)
        REQUIRED_ENV_VARS="${2:?缺少 required env 列表}"
        shift 2
        ;;
      --recommended-env)
        RECOMMENDED_ENV_VARS="${2:?缺少 recommended env 列表}"
        shift 2
        ;;
      --skip-llm-env-check)
        REQUIRE_LLM_PROVIDER="n"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "未知参数: $1"
        ;;
    esac
  done
}

validate_config() {
  case "$TEST_MODE" in
    skip|quick|full) ;;
    *) die "无效测试模式: $TEST_MODE" ;;
  esac

  case "$SYNC_MODE" in
    safe|mirror) ;;
    *) die "无效同步模式: $SYNC_MODE" ;;
  esac

  case "$USE_SYSTEMD" in
    auto|always|never) ;;
    *) die "无效 systemd 模式: $USE_SYSTEMD" ;;
  esac

  validate_yes_no "INSTALL_DEPS" "$INSTALL_DEPS"
  validate_yes_no "RUN_REMOTE_TESTS" "$RUN_REMOTE_TESTS"
  validate_yes_no "GIT_RELEASE_ENABLED" "$GIT_RELEASE_ENABLED"
  validate_yes_no "SKIP_HEALTHCHECK" "$SKIP_HEALTHCHECK"
  validate_yes_no "REQUIRE_LLM_PROVIDER" "$REQUIRE_LLM_PROVIDER"
  validate_positive_integer "HEALTHCHECK_TIMEOUT" "$HEALTHCHECK_TIMEOUT"
  validate_positive_integer "HEALTHCHECK_INTERVAL" "$HEALTHCHECK_INTERVAL"
}

ensure_local_conda() {
  if ! command -v conda >/dev/null 2>&1 && [[ -f "$LOCAL_CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$LOCAL_CONDA_SH"
  fi

  command -v conda >/dev/null 2>&1 || die "本地 conda 不可用，请确认 LOCAL_CONDA_SH 或 PATH"
}

run_local_tests() {
  local mode="$1"
  local -a target=()

  case "$mode" in
    quick)
      target=("${QUICK_TESTS[@]}")
      ;;
    full)
      target=("${FULL_TESTS[@]}")
      ;;
    *)
      die "无效测试模式: $mode"
      ;;
  esac

  log "🧪 本地测试模式: $mode"
  ensure_local_conda
  conda run -n "$ENV_NAME" python -m pytest "${target[@]}" -q
}

print_local_versions() {
  log "📍 Local branch: ${LOCAL_BRANCH:-detached}"
  log "📍 Local HEAD:   $(git rev-parse --short "$LOCAL_HEAD")"
  log "📍 ${GIT_REMOTE}/${MAIN_BRANCH}:  $(git rev-parse --short "${GIT_REMOTE}/${MAIN_BRANCH}" 2>/dev/null || echo 'unavailable')"
}

cleanup_local_bundle() {
  rm -f "$DEPLOY_BUNDLE_NAME"
}

create_deploy_bundle() {
  cleanup_local_bundle
  log "📦 生成本地部署 bundle..."
  git bundle create "$DEPLOY_BUNDLE_NAME" "$DEPLOY_BUNDLE_REF"
}

ensure_clean_worktree() {
  local dirty_output

  dirty_output="$(git status --porcelain)"
  if [[ -n "$dirty_output" ]]; then
    printf '❌ 发布前工作区必须干净，当前未提交变更:\n' >&2
    printf '%s\n' "$dirty_output" >&2
    exit 1
  fi
}

ensure_git_remote_exists() {
  git remote get-url "$GIT_REMOTE" >/dev/null 2>&1 || die "Git remote 不存在: $GIT_REMOTE"
}

git_release() {
  local remote_main_ref="${GIT_REMOTE}/${MAIN_BRANCH}"

  if [[ "$GIT_RELEASE_ENABLED" != "y" ]]; then
    log "⏭️ 跳过 Git 发布链路"
    return 0
  fi

  [[ -n "$LOCAL_BRANCH" ]] || die "当前处于 detached HEAD，无法自动推送分支和发布 main"
  ensure_clean_worktree
  ensure_git_remote_exists

  log "🔄 拉取远端分支引用（仅 fetch，不使用 git pull）..."
  run_cmd git fetch "$GIT_REMOTE" "$MAIN_BRANCH"

  if [[ "$LOCAL_BRANCH" != "$MAIN_BRANCH" ]]; then
    git merge-base --is-ancestor "$remote_main_ref" HEAD || die "当前分支未包含最新 ${remote_main_ref}，请先 rebase 或 merge 后再发布"
    log "📤 推送当前分支到 ${GIT_REMOTE}/${LOCAL_BRANCH}..."
    run_cmd git push -u "$GIT_REMOTE" "$LOCAL_BRANCH"
    log "🚀 fast-forward 发布到 ${GIT_REMOTE}/${MAIN_BRANCH}..."
    run_cmd git push "$GIT_REMOTE" "HEAD:refs/heads/${MAIN_BRANCH}"
  else
    log "📤 推送 ${MAIN_BRANCH} 到 ${GIT_REMOTE}/${MAIN_BRANCH}..."
    run_cmd git push "$GIT_REMOTE" "$MAIN_BRANCH"
  fi

  run_cmd git fetch "$GIT_REMOTE" "$MAIN_BRANCH"
}

sync_files() {
  local -a rsync_opts=(
    -az
    --itemize-changes
    --partial
    --inplace
    --exclude=.git
    --exclude=.env
    --exclude=.env.*
    --exclude=__pycache__
    --exclude=*.pyc
    --exclude=.DS_Store
    --exclude=._*
    --exclude=.venv
    --exclude=.vscode
    --exclude=.obsidian
    --exclude=.pytest_cache
    --exclude=.mypy_cache
    --exclude=.ipynb_checkpoints
    --exclude=logs
    --exclude=README.pdf
    --exclude=nodes_eval/extractor_eval/failure_analyses/_archived
    "--filter=P /.env"
    "--filter=P /.env.*"
    "--filter=P /output.log"
    "--filter=P /*.db"
    "--filter=P /*.sqlite"
    "--filter=P /*.sqlite3"
    "--filter=P /data/"
    "--filter=P /storage/"
    "--filter=P /chroma/"
    "--filter=P /faiss/"
    "--filter=P /vector_store/"
  )

  if [[ "$SYNC_MODE" == "mirror" ]]; then
    rsync_opts+=(--delete --delete-delay)
    log "🚀 镜像同步本地工作区到服务器（启用删除保护规则）..."
  else
    log "🚀 安全同步本地工作区到服务器（默认不删除远端状态文件）..."
  fi

  rsync "${rsync_opts[@]}" ./ "$SERVER_USER@$SERVER_IP:$REMOTE_PATH"
}

remote_deploy() {
  local quick_tests_csv
  quick_tests_csv="$(join_csv "${QUICK_TESTS[@]}")"

  ssh -T "$SERVER_USER@$SERVER_IP" \
    "REMOTE_PATH=$(printf '%q' "$REMOTE_PATH") \
ENV_NAME=$(printf '%q' "$ENV_NAME") \
REMOTE_CONDA_SH=$(printf '%q' "$REMOTE_CONDA_SH") \
DEPLOY_BUNDLE_NAME=$(printf '%q' "$DEPLOY_BUNDLE_NAME") \
LOCAL_HEAD=$(printf '%q' "$LOCAL_HEAD") \
INSTALL_DEPS=$(printf '%q' "$INSTALL_DEPS") \
RUN_REMOTE_TESTS=$(printf '%q' "$RUN_REMOTE_TESTS") \
USE_SYSTEMD=$(printf '%q' "$USE_SYSTEMD") \
SYSTEMD_SERVICE=$(printf '%q' "$SYSTEMD_SERVICE") \
SYSTEMD_CANDIDATES_CSV=$(printf '%q' "$SYSTEMD_CANDIDATES_CSV") \
SYSTEMD_UNIT_SOURCE_PATH=$(printf '%q' "$SYSTEMD_UNIT_SOURCE_PATH") \
REMOTE_HEALTHCHECK_URL=$(printf '%q' "$REMOTE_HEALTHCHECK_URL") \
HEALTHCHECK_TIMEOUT=$(printf '%q' "$HEALTHCHECK_TIMEOUT") \
HEALTHCHECK_INTERVAL=$(printf '%q' "$HEALTHCHECK_INTERVAL") \
SKIP_HEALTHCHECK=$(printf '%q' "$SKIP_HEALTHCHECK") \
REQUIRED_ENV_VARS=$(printf '%q' "$REQUIRED_ENV_VARS") \
RECOMMENDED_ENV_VARS=$(printf '%q' "$RECOMMENDED_ENV_VARS") \
REQUIRE_LLM_PROVIDER=$(printf '%q' "$REQUIRE_LLM_PROVIDER") \
REMOTE_LOG_FILE=$(printf '%q' "$REMOTE_LOG_FILE") \
PIP_INDEX_URL=$(printf '%q' "$PIP_INDEX_URL") \
PIP_TRUSTED_HOST=$(printf '%q' "$PIP_TRUSTED_HOST") \
QUICK_TESTS_CSV=$(printf '%q' "$quick_tests_csv") \
bash -s" <<'EOF'
set -euo pipefail

restart_mode="manual"
current_service=""

log() {
  printf '%s\n' "$1"
}

die() {
  printf '❌ %s\n' "$1" >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

env_var_is_configured() {
  local key="$1"
  local env_value="${!key:-}"
  if [[ -n "$env_value" ]]; then
    return 0
  fi

  [[ -f .env ]] || return 1
  grep -Eq "^[[:space:]]*(export[[:space:]]+)?${key}=.+$" .env
}

check_env_list() {
  local vars_csv="$1"
  local level="$2"
  local -a vars=()
  local -a missing=()
  local raw
  local var

  [[ -n "$vars_csv" ]] || return 0
  IFS=',' read -r -a vars <<< "$vars_csv"

  for raw in "${vars[@]}"; do
    var="$(trim "$raw")"
    [[ -n "$var" ]] || continue
    if ! env_var_is_configured "$var"; then
      missing+=("$var")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    return 0
  fi

  if [[ "$level" == "required" ]]; then
    die "服务器缺少必需环境变量: ${missing[*]}"
  fi

  log "⚠️ 建议补齐环境变量: ${missing[*]}"
}

env_group_is_configured() {
  local key
  for key in "$@"; do
    env_var_is_configured "$key" || return 1
  done
}

validate_llm_provider_env() {
  [[ "$REQUIRE_LLM_PROVIDER" == "y" ]] || return 0

  if env_group_is_configured DEEPSEEK_API_KEY || \
     env_group_is_configured GOOGLE_API_KEY || \
     env_group_is_configured DOUBAO_API_KEY DOUBAO_BASE_URL || \
     env_group_is_configured QWEN_API_KEY QWEN_BASE_URL || \
     env_group_is_configured DASHSCOPE_API_KEY QWEN_BASE_URL; then
    log "🔐 LLM 提供商配置校验通过"
    return 0
  fi

  die "未检测到可用的 LLM 提供商配置，请至少配置一组 DeepSeek / Gemini / Doubao / Qwen 凭证"
}

validate_wecom_env() {
  local -a wecom_vars=(WECOM_CORPID WECOM_SECRET WECOM_TOKEN WECOM_AES_KEY WECOM_KF_ID)
  local configured_count=0
  local key

  for key in "${wecom_vars[@]}"; do
    if env_var_is_configured "$key"; then
      configured_count=$((configured_count + 1))
    fi
  done

  if (( configured_count == 0 )); then
    log "⚠️ 未检测到企业微信配置，若当前部署只服务 Web 端可忽略"
    return 0
  fi

  if (( configured_count < ${#wecom_vars[@]} )); then
    log "⚠️ 企业微信配置不完整，建议补齐: ${wecom_vars[*]}"
  fi
}

backup_dirty_worktree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    local backup_dir
    backup_dir="$HOME/baoshu_ai_git_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    git status --short > "$backup_dir/status.txt"
    git diff > "$backup_dir/working_tree.patch"
    log "🧷 已备份服务器脏工作树到: $backup_dir"
  fi
}

apply_deploy_bundle() {
  [[ -f "$DEPLOY_BUNDLE_NAME" ]] || die "缺少部署 bundle: $DEPLOY_BUNDLE_NAME"
  git fetch "$DEPLOY_BUNDLE_NAME" "$LOCAL_HEAD:refs/codex/deploy"
  git reset --hard "$LOCAL_HEAD"
  git update-ref -d refs/codex/deploy || true
  rm -f "$DEPLOY_BUNDLE_NAME"
}

ensure_redis_ready() {
  command -v redis-server >/dev/null 2>&1 || die "Redis 未安装，请先完成服务器基础环境初始化"
  redis-cli ping | grep -q "PONG" || die "Redis 未运行，部署终止"
}

activate_conda_env() {
  [[ -f "$REMOTE_CONDA_SH" ]] || die "找不到远端 conda 初始化脚本: $REMOTE_CONDA_SH"
  # shellcheck disable=SC1090
  source "$REMOTE_CONDA_SH"
  conda activate "$ENV_NAME"
}

install_dependencies_if_needed() {
  if [[ "$INSTALL_DEPS" != "y" ]]; then
    return 0
  fi

  log "📦 更新依赖..."
  python -m pip install -r requirements.txt -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
}

run_remote_tests_if_needed() {
  local -a quick_tests=()

  if [[ "$RUN_REMOTE_TESTS" != "y" ]]; then
    return 0
  fi

  IFS=',' read -r -a quick_tests <<< "$QUICK_TESTS_CSV"
  log "🧪 运行服务器快速回归..."
  PYTHONPATH=. python -m pytest "${quick_tests[@]}" -q
}

find_systemd_service() {
  local -a candidates=()
  local raw
  local candidate

  command -v systemctl >/dev/null 2>&1 || return 1

  if [[ -n "$SYSTEMD_SERVICE" ]]; then
    printf '%s' "$SYSTEMD_SERVICE"
    return 0
  fi

  IFS=',' read -r -a candidates <<< "$SYSTEMD_CANDIDATES_CSV"
  for raw in "${candidates[@]}"; do
    candidate="$(trim "$raw")"
    [[ -n "$candidate" ]] || continue
    if systemctl list-unit-files --type=service --all | awk '{print $1}' | grep -Fxq "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  return 1
}

install_systemd_unit_if_present() {
  local source_path="$SYSTEMD_UNIT_SOURCE_PATH"
  local unit_name=""
  local install_path=""

  [[ -f "$source_path" ]] || return 0
  command -v systemctl >/dev/null 2>&1 || die "检测到 systemd unit 文件，但服务器没有 systemctl"

  if [[ -n "$SYSTEMD_SERVICE" ]]; then
    unit_name="$SYSTEMD_SERVICE"
  else
    unit_name="$(basename "$source_path")"
    SYSTEMD_SERVICE="$unit_name"
  fi

  install_path="/etc/systemd/system/$unit_name"
  if ! cmp -s "$source_path" "$install_path"; then
    log "🧩 安装/更新 systemd unit: $unit_name"
    cp "$source_path" "$install_path"
    chmod 644 "$install_path"
    systemctl daemon-reload
  fi

  systemctl enable "$unit_name" >/dev/null
}

restart_with_systemd() {
  local service_name="$1"

  log "🔄 通过 systemd 重启服务: $service_name"
  systemctl reset-failed "$service_name" || true
  systemctl restart "$service_name"
  systemctl is-active --quiet "$service_name" || {
    journalctl -u "$service_name" -n 50 --no-pager || true
    die "systemd 服务未进入 active 状态: $service_name"
  }

  restart_mode="systemd"
  current_service="$service_name"
}

restart_without_systemd() {
  local python_bin
  local process_pattern

  python_bin="$(command -v python)"
  process_pattern="$python_bin main.py"

  log "⚠️ 未使用 systemd，回退到兼容重启模式"
  if pgrep -f "$process_pattern" >/dev/null 2>&1; then
    pkill -TERM -f "$process_pattern" || true
    for _ in $(seq 1 20); do
      if ! pgrep -f "$process_pattern" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done

    if pgrep -f "$process_pattern" >/dev/null 2>&1; then
      log "⚠️ 进程未在宽限期内退出，执行强制停止"
      pkill -KILL -f "$process_pattern" || true
      sleep 2
    fi
  fi

  nohup "$python_bin" main.py > "$REMOTE_LOG_FILE" 2>&1 < /dev/null &
  sleep 2

  pgrep -f "$process_pattern" >/dev/null 2>&1 || {
    tail -n 20 "$REMOTE_LOG_FILE" || true
    die "服务启动失败"
  }
}

restart_service() {
  local detected_service=""

  case "$USE_SYSTEMD" in
    auto|always)
      if detected_service="$(find_systemd_service)"; then
        restart_with_systemd "$detected_service"
        return 0
      fi

      if [[ "$USE_SYSTEMD" == "always" ]]; then
        die "未找到可用的 systemd 服务，请通过 --service-name 指定"
      fi
      ;;
  esac

  restart_without_systemd
}

collect_runtime_logs() {
  local logs=""

  if [[ "$restart_mode" == "systemd" ]] && [[ -n "$current_service" ]] && command -v journalctl >/dev/null 2>&1; then
    logs="$(journalctl -u "$current_service" -n 200 --no-pager 2>/dev/null || true)"
  fi

  if [[ -z "$logs" ]] && [[ -f "$REMOTE_LOG_FILE" ]]; then
    logs="$(tail -n 200 "$REMOTE_LOG_FILE" 2>/dev/null || true)"
  fi

  printf '%s' "$logs"
}

print_runtime_logs() {
  local lines="${1:-20}"

  if [[ "$restart_mode" == "systemd" ]] && [[ -n "$current_service" ]] && command -v journalctl >/dev/null 2>&1; then
    journalctl -u "$current_service" -n "$lines" --no-pager || true
    return 0
  fi

  tail -n "$lines" "$REMOTE_LOG_FILE" || true
}

wait_for_health() {
  local elapsed=0

  if [[ "$SKIP_HEALTHCHECK" == "y" ]]; then
    log "⏭️ 跳过健康检查"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || die "服务器缺少 curl，无法执行 HTTP 健康检查"

  log "🩺 等待健康检查通过: $REMOTE_HEALTHCHECK_URL"
  while (( elapsed < HEALTHCHECK_TIMEOUT )); do
    if curl --silent --show-error --fail --max-time 5 "$REMOTE_HEALTHCHECK_URL" >/dev/null; then
      log "✅ HTTP 健康检查通过"
      return 0
    fi

    sleep "$HEALTHCHECK_INTERVAL"
    elapsed=$((elapsed + HEALTHCHECK_INTERVAL))
  done

  print_runtime_logs 30
  die "HTTP 健康检查超时，等待了 ${HEALTHCHECK_TIMEOUT}s"
}

verify_runtime_signals() {
  local logs
  local -a required_signals=(
    "LangGraph checkpointer: PostgresSaver"
    "LangGraph backend ready: postgres"
    "Postgres 连接成功，Schema 已就绪"
  )
  local signal

  if [[ "$SKIP_HEALTHCHECK" == "y" ]]; then
    return 0
  fi

  logs="$(collect_runtime_logs)"
  [[ -n "$logs" ]] || die "未能读取服务日志，无法验证运行时信号"

  for signal in "${required_signals[@]}"; do
    if ! grep -Fq "$signal" <<< "$logs"; then
      print_runtime_logs 30
      die "服务日志缺少关键启动信号: $signal"
    fi
  done
}

cd "$REMOTE_PATH"
backup_dirty_worktree
apply_deploy_bundle
ensure_redis_ready
check_env_list "$REQUIRED_ENV_VARS" required
check_env_list "$RECOMMENDED_ENV_VARS" recommended
validate_llm_provider_env
validate_wecom_env
activate_conda_env
install_dependencies_if_needed
run_remote_tests_if_needed
install_systemd_unit_if_present
restart_service
wait_for_health
verify_runtime_signals

log "✅ 服务启动成功"
log "📍 Server branch: $(git branch --show-current || true)"
log "📍 Server HEAD:   $(git rev-parse --short HEAD || true)"
log "📍 Server status: $(git status --short | wc -l | tr -d ' ') modified entries"
print_runtime_logs 15
EOF
}

main() {
  parse_args "$@"
  validate_config
  trap cleanup_local_bundle EXIT

  case "$TEST_MODE" in
    skip)
      log "⏭️ 跳过本地测试"
      ;;
    quick|full)
      run_local_tests "$TEST_MODE"
      ;;
  esac

  git_release
  print_local_versions
  create_deploy_bundle
  sync_files
  remote_deploy
  log "🎉 部署完成"
}

main "$@"
