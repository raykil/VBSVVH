import json, sys, subprocess
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(prog='python local_hvv.py', epilog="jkil@nd.edu")
parser.add_argument('-c', '--channel', default="run2_1FJ", type=str, choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
parser.add_argument('-r', '--resume' , action="store_true", help="skip datasets already present in output_<channel>/. It's fine to activate at first run.")
args = parser.parse_args()

run, nFJ = args.channel[3], args.channel[5]
out_dir = Path(f"output_{args.channel}")
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
    SAMPLES = json.load(f)

# cahce file (e.g. sample_json/cache_run2_1FJ.json) lets me to look for local copy of file list and skip the time consuming xrdfs.
cache = Path("sample_json") / f"cache_{args.channel}.json"
if cache.exists():
    FILE_DICT = {k: v for k, v in json.loads(cache.read_text()).items() if k in SAMPLES}
    print(f"Using cached file list: {cache} ({len(FILE_DICT)}/{len(SAMPLES)} dataset(s) cached)")
else:
    FILE_DICT = {}
    for k, v in SAMPLES.items():
        print(f"Fetching {k}...")
        path = f"/eos/user/r/rband/HVV2LRDF/{channel_tag}/{v['metadata']['dataset']}" # later I might move to /groups/cjessop area.
        file_names = subprocess.check_output(["xrdfs", "root://eosuser.cern.ch", "ls", path], text=True).split() # returns a list of file names is Reyer's area.
        FILE_DICT[k] = [f"root://eosuser.cern.ch//{f}" for f in file_names]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(FILE_DICT, indent=4))

if args.resume:
    done = [k for k in FILE_DICT if (out_dir / year_tag[SAMPLES[k]["metadata"]["year"]] / k / "out.pkl").exists()] # check if out.pkl is present
    FILE_DICT = {k: v for k, v in FILE_DICT.items() if k not in done}
    print(f"Skipping {len(done)} already-processed dataset(s), {len(FILE_DICT)} left")

Path("local_hvv").mkdir(exist_ok=True)
for dataset_name, files in FILE_DICT.items():
    # ————— Processing —————————————————————————
    year = year_tag[SAMPLES[dataset_name]["metadata"]["year"]]
    print(f"\nProcessing {dataset_name}: {len(files)} file(s)")
    subprocess.run(["python", "src/run.py", "-y", year, "-d", dataset_name, "-fj", nFJ, "-v", "v15", "-f", *files, "--save-skim"], check=True) # output of run.py is in local_hvv.

    # ————— Saving Output —————————————————————————
    dest = out_dir / year / dataset_name
    dest.mkdir(parents=True, exist_ok=True)
    for parq in Path("local_hvv").glob("*.parquet"):
        parq.replace(dest / f"{parq.stem}.parq")
    for pkl in Path("local_hvv").glob("*.pkl"):
        pkl.replace(dest / "out.pkl")
    print(f"Saved {len(list(dest.iterdir()))} output(s) to {dest}/")

Path("local_hvv").rmdir()
