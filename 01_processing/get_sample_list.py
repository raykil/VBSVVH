import json
from argparse import ArgumentParser
from pathlib import Path

scriptPath = Path(__file__).resolve().parent
parser = ArgumentParser(prog='python get_sample_list.py', epilog="jkil@nd.edu")
parser.add_argument('-c', '--channels', default=["run2_1FJ"], type=str, nargs='+', choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'], help='space-separated.')
args = parser.parse_args()

DATA_DIR = Path("/groups/cjessop/users/jkil/HVV_2L_RDF") # copy of Reyer's /eos/user/r/rband/HVV2LRDF: condor workers have no CERN identity for eosuser.
year_tags = {  # sample json metadata.year -> year in run.py
    '2016postVFP' : "2016",
    '2016preVFP'  : "2016APV",
    '2017'        : "2017",
    '2018'        : "2018",
    '2022'        : "2022",
    '2023'        : "2023",
    '2024Prompt'  : "2024",
    '2025'        : "2025"
}

for channel in args.channels:
    inputdir  = Path(f"{scriptPath}/sample_json/samples_{channel}.json")
    outputdir = Path(f"{scriptPath}/sample_json/files_{channel}.json")
    SAMPLES   = json.loads(inputdir.read_text())

    FILE_DICT = {}
    for k, v in SAMPLES.items():
        print(f"Fetching {k}...")
        file_names = [f for f in sorted((DATA_DIR / channel / v['metadata']['dataset']).glob("*.root")) if f.stat().st_size]
        if not file_names: # empty upstream, and coffea raises on a dataset with no usable file
            print(f"\033[1;31m  {k}: no non-empty files, dropped\033[0m")
            continue
        year = year_tags[v['metadata']['year']]

        FILE_DICT.setdefault(year, {})
        FILE_DICT[year][k] = [str(f) for f in file_names]

    outputdir.parent.mkdir(exist_ok=True)
    outputdir.write_text(json.dumps(FILE_DICT, indent=4))
    print(f"\033[1;32m{outputdir} created!\033[0m")