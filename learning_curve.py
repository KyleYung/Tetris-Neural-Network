import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import os
import re                           # regular expressions


LINE_PREAMBLE = "[INFO] TrainerAgent.onGameEnd: After "
LINE_POSTAMBLE = "phase(s), avg trajectory utility = "


def load(path: str) -> np.ndarray:
    data: List[Tuple[int, float]] = list()

    try:
        with open(path, "r") as f:
            for line in f:
                if LINE_PREAMBLE in line.strip().rstrip() and LINE_POSTAMBLE in line.strip().rstrip():
                    values_str = line.strip().rstrip().replace(LINE_PREAMBLE, "").replace(LINE_POSTAMBLE, "")
                    phase_idx, avg_utility = re.sub(r'\s+', ' ', values_str.strip().rstrip()).strip().rstrip().split(" ")
                    data.append([float(phase_idx), float(avg_utility)])
    except:
        pass

    return np.array(data)


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("logfile", type=str, help="path to logfile containing eval outputs")
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        raise Exception("ERROR: logfile [%s] does not exist!" % args.logfile)

    data: np.ndarray = load(args.logfile)
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    main()

