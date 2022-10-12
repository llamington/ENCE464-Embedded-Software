"""Script for measuring execution time and file size of different compiler optimisations"""

import subprocess
import os
import re

def main():
    flag_time = {}
    with open("O1_flags.txt") as file:
        flags = file.readlines()

    for flag in flags:
        flag = flag.strip()
        subprocess.run("make clean", shell=True)
        made = subprocess.run("make", env=dict(os.environ, OPT_FLAG=flag), shell=True)
        if made.returncode == 0:
            out = subprocess.run('hyperfine "./poisson -n 51 -i 300"', shell=True, capture_output=True).stdout
            time_line = out.decode().splitlines()[-3]
            searched = re.search("([\d.]+)", time_line)
            searched.group(0)
            os.wait(5)
            size = os.path.getsize("poisson")
            flag_time[flag] = dict(exec=float(searched.group(0)), size=size)

    with open("flag_timings.csv", "w") as file:
        file.writelines([f"{flag[2:]}, {obj['exec']}, {obj['size']}\n" for flag, obj in flag_time.items()])


if __name__ == "__main__":
    main()