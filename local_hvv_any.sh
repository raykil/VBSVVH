
#!/bin/bash

normalize_year() {
    local y="$1"

    case "$y" in
        2016postVFP) echo "2016" ;;
        2016preVFP)  echo "2016APV" ;;
        2017*)       echo "2017" ;;
        2018*)       echo "2018" ;;
        2022*)       echo "2022" ;;
        2023BPix*)   echo "2023BPix" ;;
        2023*)       echo "2023" ;;
        2024*)       echo "2024" ;;
        *)
            echo "UNKNOWN"
            ;;
    esac
}
case "$1" in
    1FJ_r3)
        process_tag="merged_2lep_1FJ_r3_2lep_1FJ_20260615211529_2lep_1FJ"
        ;;
    2FJ_r3)
        process_tag="merged_2lep_2FJ_r3_2lep_2FJ_20260615211951_2lep_2FJ"
        ;;
    2FJ_r2)
        process_tag="merged_2lep_2FJ_r2_2lep_2FJ_20260615211710_2lep_2FJ"
        ;;
    1FJ_r2)
        process_tag="merged_2lep_1FJ_r2_2lep_1FJ_20260615211230_2lep_1FJ"
        ;;
    *)
        echo "Unknown dataset: $2"
        exit 1
        ;;
esac
# remove old files
rm -r local_hvv/
mkdir local_hvv/

tag="hvv_26Aug19"
#process_tag="merged_2lep_1FJ_r3_2lep_1FJ_20260615211529_2lep_1FJ"
#process_tag="merged_2lep_2FJ_r3_2lep_2FJ_20260615211951_2lep_2FJ"
#process_tag="merged_2lep_2FJ_r2_2lep_2FJ_20260615211710_2lep_2FJ"
#process_tag="merged_2lep_1FJ_r2_2lep_1FJ_20260615211230_2lep_1FJ"
basedir="/eos/user/r/rband/HVV2LRDF/${process_tag}"
indir="${basedir}"
json_name="${process_tag%_2lep_[12]FJ}.json"
nFJ=${process_tag##*_2lep_}
nFJ=${nFJ%FJ}
echo $nFJ
#signal
# process_tag="merged_2lep_1FJ_r3_2lep_1FJ_20260504235849"
# basedir="/eos/user/r/rband/HVV2LRDF/2lep_1FJ_r3_2lep_1FJ/"
# indir="${basedir}/merged_2lep_1FJ_r3_2lep_1FJ_20260504235849_2lep_1FJ/"




#r2:

# process_tag="merged_2lep_1FJ_r2_2lep_1FJ_20260430154928"
# basedir="/eos/user/r/rband/HVV2LRDF/2lep_1FJ_r2_2lep_1FJ/"
# indir="${basedir}/merged_2lep_1FJ_r2_2lep_1FJ_20260430154928_2lep_1FJ/"
outdir="/store/user/rband/${tag}/${process_tag}"
#xrdcp "root://eosuser.cern.ch///${basedir}/${process_tag}.json" .
xrdcp "root://eosuser.cern.ch///${basedir}/${json_name}" .

count=0
#for process in $(jq -r '.samples | keys[]' "${process_tag}.json")
for process in $(jq -r '.samples | keys[]' "${json_name}")
do

    year_raw=$(jq -r --arg proc "$process" '.samples[$proc].metadata.year' "${json_name}")
    year=$(normalize_year "$year_raw")
#    if [[ "$year" == "2018" ]]; then
#	continue
#    fi
    echo "Processing sample: $process"
#    echo "Using year: $year"
    ((count++))
    # if [[ $count -ge 2 ]]; then
    #     echo "Stopping after 2 samples (test mode)"
    #     break
    # fi
    for folder in pickles parquet
    do
        xrdfs root://cmseos.fnal.gov// mkdir -p "/${outdir}/${year}/${process}/${folder}"
    done

    file_list=()
    while IFS= read -r infile
    do
    rootfile="root://eosuser.cern.ch//${infile}"

    # Get number of entries in the Events tree

    nentries=$(python - "$rootfile" <<'PY'
import sys
import ROOT

f = ROOT.TFile.Open(sys.argv[1])
if not f or f.IsZombie():
    print(-1)
else:
    t = f.Get("Events")
    print(t.GetEntries() if t else -1)
    f.Close()
PY
)


    if [[ "$nentries" =~ ^[0-9]+$ ]] && [[ "$nentries" -gt 0 ]]; then
        echo "Adding: $rootfile ($nentries events)"
        file_list+=("$rootfile")
    else
        echo "Skipping empty/unreadable file: $rootfile (entries=$nentries)"
    fi   
    done < <(xrdfs root://eosuser.cern.ch ls "${indir}/${process}")
    # xrdfs root://eosuser.cern.ch ls "${indir}/${process}" | while IFS= read -r infile 
    # do
    # Skip this process if no non-empty ROOT files were found
    if [[ ${#file_list[@]} -eq 0 ]]; then
        echo "No non-empty ROOT files found for ${process}, skipping."
        continue
    fi

        echo "Processing: $infile"

        # base=$(basename "$infile") 
        # jobnum=${base#output_}
        # jobnum=${jobnum%.root} 

        jobnum="0"

        # Example: pass to your command
        # python src/run.py --year "${year}" --nano-version v15 --save-skim  --files root://eosuser.cern.ch//${infile}
        python src/run.py --year "${year}" --nano-version v15 --save-skim  --files "${file_list[@]}" --dataset "${process}" --nFJ ${nFJ}


        # Move final output to EOS
        # This new logic recursively copies the region directories created by the processor

        # --- FINAL COPY LOGIC ---
        # This logic creates the nested structure and partN.parquet names

        xrdfs root://cmseos.fnal.gov// mkdir -p "/${outdir}/${year}/${process}/pickles"
        xrdcp -f local_hvv/*.pkl "root://cmseos.fnal.gov///${outdir}/${year}/${process}/pickles/out_${jobnum}.pkl"

        # 2. Next, handle the combined parquet files
        for file in local_hvv/*.parquet; do
            echo $file
            # Extract the region name from the local filename (e.g., gets "control-tt" from "control-tt.parquet")
            base_file=$(basename "${file}" ".parquet")
            region_name="${base_file##*_}"
            jer_name="${base_file%_*}"

            # Create the region-specific subdirectory on EOS
            xrdfs root://cmseos.fnal.gov// mkdir -p "/${outdir}/${year}/${process}/parquet/${jer_name}/${region_name}"

            # Define the final filename using the job number for uniqueness
            final_filename="part${jobnum}.parquet"

            # Copy the file to its final, nested destination with the new name
#            echo "root://cmseos.fnal.gov///${outdir}/${year}/${process}/parquet/${jer_name}/${region_name}/${final_filename}"
            xrdcp -f "$file" "root://cmseos.fnal.gov///${outdir}/${year}/${process}/parquet/${jer_name}/${region_name}/${final_filename}"
        done


        rm local_hvv/*.parquet
        rm local_hvv/*.pkl


    # done
    


done
