"""
Runs per channel per year. All datasets of that year go into one fileset.
workers return the per-region arrays and this
script writes the parquet.
"""
import argparse, json, os, pickle, warnings, dask
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*") # must precede the coffea import.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*reduce_op.*")

import awkward as ak
from pathlib import Path
from coffea.dataset_tools import apply_to_fileset, preprocess
from ndcctools.taskvine import DaskVine
from utils.rdf_schema import RDFSchema
from utils.categorizer import categorizer

scriptPath = Path(__file__).resolve().parent

def run(year, nFJ, fileset, args):
    outdir = (scriptPath / f"../results/{args.channel}/{year}").resolve()
    files = {d: {"files": dict.fromkeys(f, "Events"), "metadata": {"dataset": d}} for d, f in fileset.items()}
    uproot_options = {"allow_read_errors_with_report": OSError, "timeout": 1800} # KeyError left uncaught: a missing branch should crash, not silently empty the output.

    # —————————— TaskVine manager —————————————————————————
    manager = DaskVine([9123, 9130], name=f"{os.environ['USER']}-{args.channel}_{year}", run_info_path="/groups/cjessop/users/jkil/vine-run-info")
    print(f"TaskVine manager: {manager.name}")
    print("Run the following on condorfe:")
    print(f"\033[1;34msh run_factory.sh -M {manager.name} --wrapper ~/with_oasis_certs --poncho-env vbsvvh-env.tar.gz --cores {args.cores} --memory {args.memory} --disk 100 --workers {args.workers}\033[0m")

    # —————————— Preprocess ———————————————————————————————
    available = preprocess(
        files,
        files_per_batch = 1,
        align_clusters  = True,
        skip_bad_files  = True,
        step_size       = 20_000, # nEvents per chunk
        uproot_options  = uproot_options
    )[0]

    # —————————— Running the processor ————————————————————
    processor = categorizer(
        year = year,
        nFJ  = nFJ,
    )
    task_graph, report = apply_to_fileset(
        data_manipulation = processor,
        fileset           = available,
        schemaclass       = RDFSchema,
        uproot_options    = uproot_options
    ) # task_graph is process's output.
    modules = sorted(Path(f"{scriptPath}/utils").glob("*.py"))
    OUTPUT, REPORT = dask.compute(
        task_graph,
        report,
        scheduler = manager.get,
        resources = {"cores": 1},
        resources_mode = "fixed",
        environment = f"{scriptPath}/vbsvvh-env.tar.gz",
        extra_files = {manager.declare_file(str(m)): f"utils/{m.name}" for m in modules},
        env_vars    = {"PYTHONPATH": "."}
    )
    # format of OUTPUT:
    # OUTPUT = {
    #   dataset: {                  # e.g. "VBSWWH_OS_c2v1p0_c3_10p0_2024"
    #     "nominal": {              # only shift for now
    #       region: {               # "1FJ_WWH", "1FJ_VZH", "2FJ_VZH"
    #         "events":  ak.Array,  # selected events, 16 fields (below)
    #         "cutflow": Hist,      # Integer "cut" axis, Weight storage
    #       }
    #     }
    #   }
    # }

    # —————————— Read failures ————————————————————————————
    for dataset, rep in REPORT.items():
        for r in rep[~ak.is_none(rep.exception)]:
            print(f"\033[1;31m{dataset} on {r.hostname}: {r.exception}: {r.message}\n  {r.args[0]}\033[0m")

    # —————————— Saving results ———————————————————————————
    regions = {region for shifts in OUTPUT.values() for out in shifts.values() for region in out}
    for region in regions:
        for dataset, shifts in OUTPUT.items():
            for shift, out in shifts.items():
                path = Path(f"{outdir}/{region}/{shift}/{dataset}.parq")
                path.parent.mkdir(parents=True, exist_ok=True)
                ak.to_parquet(out[region].pop("events"), path)
        
        cutflows = {dataset: {shift: out[region] for shift, out in shifts.items()} for dataset, shifts in OUTPUT.items()} # events popped above
        (outdir / region / "cutflow.pkl").write_bytes(pickle.dumps(cutflows)) # next to the shift dirs: {dataset: {shift: {"cutflow", "btagWeight", "filltime"}}}
    
    print(f"\033[1;32m{outdir.relative_to(scriptPath.parent)} written!\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c' , "--channel"     , type=str, default="run2_1FJ", choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
    parser.add_argument('-y' , "--year"        , type=str, default="2023"    , choices=["2016APV", "2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix", "2024", "2025"])

    parser.add_argument('-w' , "--workers"     , type=int, default=100       , help="pilot workers the factory should request (printed, not enforced here).")
    parser.add_argument('-m' , "--memory"      , type=int, default=10        , help="memory (GB) per worker.")
    parser.add_argument('-n' , "--cores"       , type=int, default=4         , help="cores per worker.")
    args = parser.parse_args()

    filepath = scriptPath/f"sample_json/files_{args.channel}.json"
    fileset = json.loads(filepath.read_text())[args.year] # dict in files_*.json.
    run(args.year, int(args.channel[5]), fileset, args)
