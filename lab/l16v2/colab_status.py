# Run on the VM by the driver (colab exec -f): the job's status line, its markers with the product each one names,
# its finished-run lines. Bytes in, ASCII out, so nothing a log holds can break the read on either side.
import subprocess
raw = subprocess.run(
    "echo STATUS $(cat /content/l16v2/status.txt 2>/dev/null); "
    "for m in /content/l16v2/markers/*; do [ -e \"$m\" ] && echo MARK $(basename \"$m\") $(head -n 1 \"$m\"); done; "
    "echo FINISHED_BEGIN; cat /content/l16v2/finished.txt 2>/dev/null; echo FINISHED_END; "
    "echo TIMES $(cat /content/l16v2/times.txt 2>/dev/null | tr '\n' ';'); "
    "echo GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null); "
    "echo ALIVE $(pgrep -f '[c]olab_job.sh' | wc -l); echo UPTIME $(cut -d' ' -f1 /proc/uptime)",
    shell=True, capture_output=True).stdout
print(raw.decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"))
