#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DB_NAME:-baoshu_ai}"
DB_USER="${DB_USER:-baoshu_ai}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_PORT="${DB_PORT:-5432}"

if [[ -z "$DB_PASSWORD" ]]; then
  echo "DB_PASSWORD is required"
  exit 1
fi

run_psql() {
  runuser -u postgres -- psql "$@"
}

configure_auth() {
  local hba_file
  hba_file="$(run_psql -Atc "SHOW hba_file" postgres)"

  if [[ -n "$hba_file" && -f "$hba_file" ]]; then
    sed -i 's/^host[[:space:]]\+all[[:space:]]\+all[[:space:]]\+127\.0\.0\.1\/32[[:space:]]\+ident$/host    all             all             127.0.0.1\/32            scram-sha-256/' "$hba_file"
    sed -i 's/^host[[:space:]]\+all[[:space:]]\+all[[:space:]]\+::1\/128[[:space:]]\+ident$/host    all             all             ::1\/128                 scram-sha-256/' "$hba_file"
    systemctl reload postgresql
  fi
}

install_postgres() {
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y postgresql-server postgresql-contrib >&2
    if [[ ! -d /var/lib/pgsql/data/base ]]; then
      postgresql-setup --initdb >&2
    fi
  elif command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update >&2
    apt-get install -y postgresql postgresql-contrib >&2
  else
    echo "Unsupported package manager"
    exit 1
  fi

  systemctl enable postgresql >&2
  systemctl restart postgresql >&2
}

install_postgres
configure_auth

run_psql -tc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER'" | grep -q 1 || \
  run_psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" >/dev/null

run_psql -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
  run_psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" >/dev/null

run_psql -c "ALTER ROLE $DB_USER SET client_encoding TO 'UTF8';" >/dev/null
run_psql -c "ALTER ROLE $DB_USER SET timezone TO 'Asia/Shanghai';" >/dev/null

echo "DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@127.0.0.1:$DB_PORT/$DB_NAME"
