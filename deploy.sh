#!/usr/bin/env bash
set -euo pipefail

SERVER_IP="${SERVER_IP:-129.211.217.58}"
SERVER_USER="${SERVER_USER:-root}"
REMOTE_PATH="${REMOTE_PATH:-/home/ubuntu/baoshu_ai}"
ENV_NAME="${ENV_NAME:-agent}"
REMOTE_CONDA_SH="${REMOTE_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
LOCAL_CONDA_SH="${LOCAL_CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
DEPLOY_BUNDLE_NAME="${DEPLOY_BUNDLE_NAME:-.codex_deploy.bundle}"

LOCAL_BRANCH="$(git branch --show-current)"
LOCAL_HEAD="$(git rev-parse HEAD)"

QUICK_TESTS=(
  tests/test_profile_state.py
  tests/test_nodes_unit.py
  tests/test_extractor_eval.py
  tests/test_intent_guardrails.py
)

FULL_TESTS=(tests)

run_local_tests() {
  local mode="$1"
  local -a target=()

  if [[ "$mode" == "quick" ]]; then
    target=("${QUICK_TESTS[@]}")
  else
    target=("${FULL_TESTS[@]}")
  fi

  echo "🧪 本地测试模式: $mode"

  if [[ -f "$LOCAL_CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$LOCAL_CONDA_SH"
    conda activate "$ENV_NAME"
  fi

  PYTHONPATH=. python -m pytest "${target[@]}" -q
}

print_local_versions() {
  echo "📍 Local branch: $LOCAL_BRANCH"
  echo "📍 Local HEAD:   $(git rev-parse --short "$LOCAL_HEAD")"
  echo "📍 Origin/main:  $(git rev-parse --short origin/main 2>/dev/null || echo 'unavailable')"
}

cleanup_local_bundle() {
  rm -f "$DEPLOY_BUNDLE_NAME"
}

create_deploy_bundle() {
  cleanup_local_bundle
  echo "📦 生成本地部署 bundle..."
  git bundle create "$DEPLOY_BUNDLE_NAME" "$LOCAL_BRANCH"
}

sync_files() {
  echo "🚀 同步本地工作区到服务器..."
  rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '._*' \
    --exclude '.venv' \
    --exclude '.vscode' \
    --exclude '.obsidian' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    --exclude '.ipynb_checkpoints' \
    --exclude 'logs' \
    --exclude 'README.pdf' \
    --exclude 'nodes_eval/extractor_eval/failure_analyses/_archived' \
    ./ "$SERVER_USER@$SERVER_IP:$REMOTE_PATH"
}

remote_deploy() {
  local install_deps="$1"
  local run_remote_tests="$2"

  ssh -T "$SERVER_USER@$SERVER_IP" <<EOF
set -euo pipefail

cd "$REMOTE_PATH"

if [[ ! -f "$DEPLOY_BUNDLE_NAME" ]]; then
  echo "❌ 缺少部署 bundle: $DEPLOY_BUNDLE_NAME"
  exit 1
fi

if [[ -n "\$(git status --porcelain)" ]]; then
  backup_dir="\$HOME/baoshu_ai_git_backup_\$(date +%Y%m%d_%H%M%S)"
  mkdir -p "\$backup_dir"
  git status --short > "\$backup_dir/status.txt"
  git diff > "\$backup_dir/working_tree.patch"
  echo "🧷 已备份服务器脏工作树到: \$backup_dir"
fi

git fetch "$DEPLOY_BUNDLE_NAME" "$LOCAL_BRANCH:refs/heads/$LOCAL_BRANCH"
git checkout -B "$LOCAL_BRANCH" "$LOCAL_HEAD"
git reset --hard "$LOCAL_HEAD"
rm -f "$DEPLOY_BUNDLE_NAME"

if ! command -v redis-server >/dev/null 2>&1; then
  echo "❌ Redis 未安装，请先完成服务器基础环境初始化。"
  exit 1
fi

if ! redis-cli ping | grep -q "PONG"; then
  echo "❌ Redis 未运行，部署终止。"
  exit 1
fi

source "$REMOTE_CONDA_SH"
conda activate "$ENV_NAME"

if [[ "$install_deps" == "y" ]]; then
  echo "📦 更新依赖..."
  pip install -r requirements.txt -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com
fi

if [[ "$run_remote_tests" == "y" ]]; then
  echo "🧪 运行服务器快速回归..."
  PYTHONPATH=. python -m pytest \
    tests/test_profile_state.py \
    tests/test_nodes_unit.py \
    tests/test_extractor_eval.py \
    tests/test_intent_guardrails.py \
    -q
fi

echo "🔄 重启服务..."
pkill -f '/root/miniconda3/envs/$ENV_NAME/bin/python main.py' || true
nohup /root/miniconda3/envs/$ENV_NAME/bin/python main.py > output.log 2>&1 < /dev/null &
sleep 3

if ! pgrep -f '/root/miniconda3/envs/$ENV_NAME/bin/python main.py' >/dev/null; then
  echo "❌ 服务启动失败"
  tail -n 20 output.log
  exit 1
fi

echo "✅ 服务启动成功"
echo "📍 Server branch: \$(git branch --show-current || true)"
echo "📍 Server HEAD:   \$(git rev-parse --short HEAD || true)"
echo "📍 Server status: \$(git status --short | wc -l | tr -d ' ') modified entries"
tail -n 10 output.log
EOF
}

echo "是否运行本地测试？[skip/quick/full]"
read -r -p "[Default: quick]: " test_mode
test_mode="${test_mode:-quick}"

case "$test_mode" in
  skip)
    echo "⏭️ 跳过本地测试"
    ;;
  quick|full)
    run_local_tests "$test_mode"
    ;;
  *)
    echo "❌ 无效测试模式: $test_mode"
    exit 1
    ;;
esac

echo "是否在服务器上安装/更新依赖？(y/n)"
read -r -p "[Default: n]: " install_deps
install_deps="${install_deps:-n}"

echo "是否运行服务器快速回归？(y/n)"
read -r -p "[Default: y]: " run_remote_tests
run_remote_tests="${run_remote_tests:-y}"

print_local_versions
create_deploy_bundle
sync_files
remote_deploy "$install_deps" "$run_remote_tests"
cleanup_local_bundle

echo "🎉 部署完成"
