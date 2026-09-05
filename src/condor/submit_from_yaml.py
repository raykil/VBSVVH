"""
Runs the submit script but with samples specified in a yaml file.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""


import argparse
from pathlib import Path

import submit
import yaml

from hbb import run_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_args(parser)
    submit.parse_args(parser)
    parser.add_argument("--yaml", default="", help="yaml file", type=str)
    args = parser.parse_args()

    with Path(args.yaml).open() as file:
        samples_to_submit = yaml.safe_load(file)

    for key, tdict in samples_to_submit.items():
        if key != args.year:
            continue
        print(f"Submitting for year {key}")
        for sample, sdict in tdict.items():
            args.samples = [sample]
            subsamples = sdict.get("subsamples", [])
            files_per_job = sdict["files_per_job"]
            if isinstance(files_per_job, dict):
                for subsample in subsamples:
                    args.subsamples = [subsample]
                    args.files_per_job = files_per_job[subsample]
                    print(args)
                    submit.main(args)
            else:
                args.subsamples = subsamples
                args.files_per_job = files_per_job
                print(args)
                submit.main(args)
