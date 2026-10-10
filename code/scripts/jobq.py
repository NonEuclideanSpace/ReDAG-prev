"""GPU job queue for one machine: the next free allowed GPU takes the next unfinished line of a queue file.
A line is a shell command run from code/ with CUDA_VISIBLE_DEVICES=<gpu>; "{gpu}" in the line is replaced. Lines
may be appended while the queue runs (append only; the file is re-read at every poll). A line "BARRIER" makes
the lines after it wait until every line above has finished (done or failed). Progress is kept in
<queue>.state.jsonl, so a restarted queue skips lines that finished with rc 0 and re-runs failed or killed ones
(every draw/edit/loop script resumes per prompt, so a re-run only redoes the unfinished prompts).
  python scripts/jobq.py --queue queue/lab_server.txt --kill-at 06:50       # lab server: stop everything before 07:00
  python scripts/jobq.py --queue queue/rented_node.txt --gpus 1,2,3,4         # GPU 0 serves the evaluator
  touch queue/<name>.stop                                                   # graceful stop from another shell
--kill-at takes HH:MM (local time, today or tomorrow) or +<seconds>. At the deadline running tasks get SIGTERM,
then SIGKILL after 30 s, and are recorded as killed."""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

from common import P0

from draftdag.preflight import allowed_gpus, load_machine

ap = argparse.ArgumentParser()
ap.add_argument("--queue", required=True, help="queue file (relative to the current directory, else under P0)")
ap.add_argument("--gpus", default=None, help="comma list; default: allowed GPUs of configs/machine.json for this host")
ap.add_argument("--per-gpu", type=int, default=1, help="tasks per GPU at once")
ap.add_argument("--kill-at", default=None, help="HH:MM or +seconds")
ap.add_argument("--poll", type=float, default=5.0)
ap.add_argument("--max-lines", type=int, default=10 ** 9)
ap.add_argument("--wait", action="store_true", help="keep polling for appended lines instead of exiting when the queue is empty")
args = ap.parse_args()

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # commands run from code/
qpath = os.path.abspath(args.queue) if (os.path.isabs(args.queue) or os.path.exists(args.queue)) else os.path.join(P0, args.queue)
name = os.path.splitext(os.path.basename(qpath))[0]
state_path = qpath + ".state.jsonl"
stop_path = os.path.join(os.path.dirname(qpath), name + ".stop")
log_dir = os.path.join(P0, "logs")
os.makedirs(log_dir, exist_ok=True)

if args.gpus:
    gpus = [int(x) for x in args.gpus.split(",")]
else:
    gpus, _dis, _host = allowed_gpus(load_machine())
if not gpus:
    raise SystemExit("no GPUs: pass --gpus or write configs/machine.json (scripts/machine_init.py)")
slots = [g for g in gpus for _ in range(max(1, args.per_gpu))]


def deadline_from(spec):
    if not spec:
        return None
    if spec.startswith("+"):
        return time.time() + float(spec[1:])
    hh, mm = [int(x) for x in spec.split(":")]
    now = dt.datetime.now()
    d = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if d <= now:
        d += dt.timedelta(days=1)
    return d.timestamp()


deadline = deadline_from(args.kill_at)


def read_queue():
    if not os.path.exists(qpath):
        return []
    out = []
    for i, line in enumerate(open(qpath, encoding="utf-8")):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append((i, s))
    return out[: args.max_lines]


def read_state():
    st = {}
    if os.path.exists(state_path):
        for line in open(state_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                st[r["line"]] = r
            except Exception:
                pass
    return st


def record(**r):
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


running = {}  # slot index -> dict(proc, line, cmd, gpu, start, log)
claimed = set()
t_start = time.time()


def launch(slot, line, cmd):
    gpu = slots[slot]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), P0=P0)
    log = os.path.join(log_dir, f"queue_{name}_L{line}_gpu{gpu}.log")
    full = cmd.replace("{gpu}", str(gpu))
    f = open(log, "a", encoding="utf-8")
    f.write(f"# {time.strftime('%F %T')} gpu {gpu}: {full}\n")
    f.flush()
    p = subprocess.Popen(full, shell=True, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    running[slot] = dict(proc=p, line=line, cmd=full, gpu=gpu, start=time.time(), log=log, fh=f)
    record(line=line, cmd=full, gpu=gpu, status="running", start=time.strftime("%F %T"), log=log)
    print(f"[{time.strftime('%T')}] start L{line} on gpu {gpu}: {full}", flush=True)


def finish(slot, status=None):
    r = running.pop(slot)
    rc = r["proc"].returncode
    r["fh"].close()
    st = status or ("done" if rc == 0 else "failed")
    record(line=r["line"], cmd=r["cmd"], gpu=r["gpu"], status=st, rc=rc, start=time.strftime("%F %T", time.localtime(r["start"])), end=time.strftime("%F %T"), sec=round(time.time() - r["start"], 1), log=r["log"])
    print(f"[{time.strftime('%T')}] {st} L{r['line']} rc={rc} after {round(time.time() - r['start'])}s", flush=True)


def kill_all(reason):
    for slot, r in list(running.items()):
        try:
            os.killpg(os.getpgid(r["proc"].pid), signal.SIGTERM)
        except Exception:
            pass
    t0 = time.time()
    while running and time.time() - t0 < 30:
        for slot, r in list(running.items()):
            if r["proc"].poll() is not None:
                finish(slot, status="killed")
        time.sleep(1)
    for slot, r in list(running.items()):
        try:
            os.killpg(os.getpgid(r["proc"].pid), signal.SIGKILL)
        except Exception:
            pass
        r["proc"].wait()
        finish(slot, status="killed")
    print(f"[{time.strftime('%T')}] stopped: {reason}", flush=True)


try:
    idle_polls = 0
    while True:
        for slot, r in list(running.items()):
            if r["proc"].poll() is not None:
                finish(slot)
        if deadline and time.time() >= deadline:
            kill_all("deadline " + args.kill_at)
            break
        if os.path.exists(stop_path):
            kill_all("stop file " + stop_path)
            os.remove(stop_path)
            break
        state = read_state()
        pending = []
        for ln, cmd in read_queue():  # a BARRIER line waits for everything above it to finish (done or failed)
            if cmd == "BARRIER":
                above_open = [l for l, c in read_queue() if l < ln and c != "BARRIER" and state.get(l, {}).get("status") not in ("done", "failed")]
                if above_open or running:
                    break
                continue
            if state.get(ln, {}).get("status") != "done" and ln not in claimed:
                pending.append((ln, cmd))
        free = [s for s in range(len(slots)) if s not in running]
        while free and pending:
            ln, cmd = pending.pop(0)
            claimed.add(ln)
            launch(free.pop(0), ln, cmd)
        if not running and not pending and not args.wait:
            idle_polls += 1
            if idle_polls >= 3:
                print(f"[{time.strftime('%T')}] queue empty, exiting", flush=True)
                break
        else:
            idle_polls = 0
        time.sleep(args.poll)
except KeyboardInterrupt:
    kill_all("keyboard interrupt")
    sys.exit(1)
