#!/usr/bin/env python3
"""每日数据更新 + 预测一键脚本（VPN-resilient，~0 token）。

流程：查 dolt 最新日期 → 算待补交易日 → curl 探测可用性 → 增量更新 → dump
→ 替换 qlib 数据 → 重生成 csi 成分股 → 跑预测 → 打印 Top10。

跑法：
    python update-predict.py             # 全自动（有新数据才更新+预测）
    python update-predict.py --predict-only   # 跳过数据更新，只跑预测

失败时把 /tmp/update_predict.log 末尾贴给 /update-predict skill 让 agent 排查。
"""
import argparse, subprocess, gzip, json, os, sys, time, glob, shutil
from pathlib import Path
import pandas as pd

# ===== 配置 =====
QLIB_ASSISTANT_DIR = "/Users/hmax/qlibAssistant"
INVESTMENT_DATA_DIR = "/Users/hmax/investment_data"
QLIB_REPO_DIR = "/Users/hmax/qlib"
QLIB_DATA_DIR = Path.home() / ".qlib/qlib_data/cn_data"
QLIBASSISTANT_PY = "/Users/hmax/miniconda3/envs/qlibAssistant/bin/python"
QLIB_ENV_PY = "/Users/hmax/miniconda3/envs/qlib_env/bin/python"
TUSHARE_URL = "https://fastapic.stockai888.top"
TOKEN_FILE = Path.home() / ".config/tushare_token"
BRANCH = "local/hmax-fixes"
LOG_FILE = "/tmp/update_predict.log"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    if os.environ.get("TUSHARE"):
        return os.environ["TUSHARE"]
    sys.exit(f"ERROR: 找不到 tushare token，写入 {TOKEN_FILE} 或设 TUSHARE 环境变量")


def tushare_curl(api_name, params, token):
    """curl 调用 tushare 代理（绕过 VPN 下 Python TLS 失败），返回 data 字段。"""
    payload = json.dumps({"api_name": api_name, "token": token, "params": params})
    r = subprocess.run(
        ["curl", "-sS", "--max-time", "60", "-X", "POST", TUSHARE_URL,
         "-H", "Content-Type: application/json", "-H", "Accept-Encoding: gzip",
         "--data", payload],
        capture_output=True, check=True,
    )
    raw = r.stdout
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    if data.get("code") != 0:
        raise Exception(f"tushare {api_name} error: {data.get('msg')}")
    return data["data"]


def dolt_sql(query):
    """dolt sql -r csv（CLI，无需 server），返回 stdout 文本。"""
    r = subprocess.run(
        ["dolt", "sql", "-r", "csv", "-q", query],
        cwd=INVESTMENT_DATA_DIR, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise Exception(f"dolt sql failed: {r.stderr.strip()}")
    return r.stdout


def run_logged(name, cmd, cwd=None, env=None):
    """运行命令，stdout/stderr 写入 LOG_FILE；失败则打印日志末尾并退出。"""
    log(f"{name} ...")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n===== {name} =====\n")
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = subprocess.run(["tail", "-30", LOG_FILE], capture_output=True, text=True).stdout
        sys.exit(f"ERROR: {name} 失败（见 {LOG_FILE}）：\n{tail}")
    log(f"{name} 完成")


def check_branches():
    log("检查仓库分支...")
    for repo in [QLIB_REPO_DIR, INVESTMENT_DATA_DIR]:
        cur = subprocess.run(
            ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
        if cur != BRANCH:
            sys.exit(f"ERROR: {repo} 在 {cur} 分支，需切到 {BRANCH}（修复才生效）")
    log(f"分支 OK（都在 {BRANCH}）")


def detect_dates(token):
    """返回待补日期列表（dolt 没有、tushare 已有数据）。"""
    out = dolt_sql("SELECT MAX(tradedate) AS d FROM final_a_stock_eod_price")
    latest = out.strip().splitlines()[-1]
    log(f"dolt 最新: {latest}")

    out = dolt_sql(
        f"SELECT date FROM ts_trade_day_calendar WHERE exchange='SSE' AND is_open=1 "
        f"AND date > '{latest}' AND date <= CURRENT_DATE ORDER BY date"
    )
    candidates = [l.strip().replace("-", "") for l in out.strip().splitlines()[1:] if l.strip()]
    log(f"候选交易日: {candidates}")

    dates = []
    for d in candidates:
        try:
            n = len(tushare_curl("daily", {"trade_date": d}, token)["items"])
        except Exception as e:
            log(f"  {d}: 探测失败 {e}")
            n = 0
        log(f"  候选 {d}: {n} 行")
        if n > 0:
            dates.append(d)
        time.sleep(0.6)
    return latest, dates


def update_data(token, dates):
    env = os.environ.copy()
    env["TUSHARE"] = token
    env["UPDATE_DATES"] = ",".join(dates)
    run_logged("增量更新 incremental_update.py",
               [QLIBASSISTANT_PY, "incremental_update.py"], cwd=INVESTMENT_DATA_DIR, env=env)


def dump_and_replace():
    env = os.environ.copy()
    env["CLEAN_QLIB_BUILD_ROOT"] = "0"
    env["PATH"] = f"/Users/hmax/miniconda3/envs/qlib_env/bin:{env['PATH']}"
    run_logged("转 qlib 二进制 dump_qlib_bin.sh",
               ["bash", "dump_qlib_bin.sh"], cwd=INVESTMENT_DATA_DIR, env=env)

    build_root = sorted(glob.glob("/Users/hmax/qlib_build_*"))[-1]
    cal_latest = (Path(build_root) / "qlib_bin/calendars/day.txt").read_text().strip().splitlines()[-1]
    log(f"qlib_bin 日历最新: {cal_latest}")

    log(f"替换 {QLIB_DATA_DIR} ...")
    subprocess.run(["rsync", "-a", "--delete", f"{build_root}/qlib_bin/", f"{QLIB_DATA_DIR}/"], check=True)

    # 重新生成 csi 成分股（rsync --delete 会删掉）
    log("重新生成 csi 成分股...")
    server = subprocess.Popen(["dolt", "sql-server"], cwd=INVESTMENT_DATA_DIR,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    try:
        env3 = os.environ.copy()
        env3["QLIB_INDEX_DIR"] = "/tmp/qlib_index"
        subprocess.run([QLIB_ENV_PY, "qlib/dump_index_weight.py"],
                       cwd=INVESTMENT_DATA_DIR, env=env3, check=True)
        for f in glob.glob("/tmp/qlib_index/csi*.txt"):
            shutil.copy(f, QLIB_DATA_DIR / "instruments")
    finally:
        server.terminate()
        server.wait()
    log(f"数据替换完成，日历到 {cal_latest}")


def run_predict():
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    run_logged("跑预测 model selection",
               [QLIBASSISTANT_PY, "./roll.py", "model", "selection"],
               cwd=f"{QLIB_ASSISTANT_DIR}/roll", env=env)


def report():
    latest_sel = sorted(glob.glob(f"{QLIB_ASSISTANT_DIR}/.qlibAssistant/analysis/selection_*"))[-1]
    files = os.listdir(latest_sel)
    ret = [f for f in files if f.endswith("_ret.csv") and "filter" not in f][0]
    filt = [f for f in files if "filter_ret" in f][0]
    df_ret = pd.read_csv(f"{latest_sel}/{ret}")
    df = pd.read_csv(f"{latest_sel}/{filt}")
    cols = [c for c in ["instrument", "name", "avg_score", "pos_ratio"] if c in df.columns]
    print(f"\n===== 预测日 {df_ret.iloc[0]['datetime']} | 全量 {len(df_ret)} 只 | 过滤后 {len(df)} 只 =====")
    print(df[cols].sort_values("avg_score", ascending=False).head(10).to_string(index=False))
    log(f"输出目录: {latest_sel}")


def main():
    parser = argparse.ArgumentParser(description="每日数据更新 + 预测")
    parser.add_argument("--predict-only", action="store_true", help="跳过数据更新，只跑预测")
    args = parser.parse_args()

    # 清空日志
    open(LOG_FILE, "w").close()
    token = get_token()

    if args.predict_only:
        log("--predict-only：跳过数据更新")
    else:
        check_branches()
        latest, dates = detect_dates(token)
        if not dates:
            log("没有可补的新数据（tushare 当天数据可能未出）。退出。")
            log("如需用现有数据跑预测：python update-predict.py --predict-only")
            return
        log(f"将更新: {','.join(dates)}")
        update_data(token, dates)
        dump_and_replace()

    run_predict()
    report()
    log("全部完成！")


if __name__ == "__main__":
    main()
