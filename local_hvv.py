import json, sys, subprocess
from argparse import ArgumentParser
parser = ArgumentParser(prog='python local_hvv.py', epilog="jkil@nd.edu")
parser.add_argument('-c', '--channel', default="run2_1FJ", type=str, choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
args = parser.parse_args()

run, nFJ = args.channel[3], args.channel[5]
channel_tag = {
    'run2_1FJ': "merged_2lep_1FJ_r2_2lep_1FJ_20260615211230_2lep_1FJ",
    'run2_2FJ': "merged_2lep_2FJ_r2_2lep_2FJ_20260615211710_2lep_2FJ",
    'run3_1FJ': "merged_2lep_1FJ_r3_2lep_1FJ_20260615211529_2lep_1FJ",
    'run3_2FJ': "merged_2lep_2FJ_r3_2lep_2FJ_20260615211951_2lep_2FJ"
} ; channel_tag = channel_tag[args.channel]
year_tag = {  # sample json metadata.year -> year in run.py
    '2016postVFP' : "2016",
    '2016preVFP'  : "2016APV",
    '2017'        : "2017",
    '2018'        : "2018",
    '2022'        : "2022",
    '2023'        : "2023",
    '2024Prompt'  : "2024"
}

with open(f"sample_json/samples_run{run}_2L_{nFJ}FJ.json", 'r') as f:
    SAMPLES = json.load(f)["samples"]

FILE_DICT = {}
for k in SAMPLES.keys():
    path = f"/eos/user/r/rband/HVV2LRDF/{channel_tag}/{k}"
    file_names = subprocess.check_output(["xrdfs", "root://eosuser.cern.ch", "ls", path], text=True).split() # returns a list of file names is Reyer's area.
    FILE_DICT[k] = [f"root://eosuser.cern.ch//{f}" for f in file_names]

for dataset_name, files in FILE_DICT.items():
    year = year_tag[SAMPLES[dataset_name]["metadata"]["year"]]
    print(f"\nProcessing {dataset_name} ({year}): {len(files)} file(s)")
    subprocess.run(["python", "src/run.py", "-y", year, "-d", dataset_name, "-fj", nFJ, "-v", "v15", "-f", *files, "--save-skim"], check=True)

    # ————— Saving Output —————————————————————————