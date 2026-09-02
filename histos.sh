#!/bin/bash

# —————————— Make Histograms ————————————————————————————————————————
for channel in run2_1FJ ; do           # run2_1FJ run2_2FJ
    for year in 2016 ; do              # 2016APV 2016 2017 2018
        for region in signal_wwh ; do  # signal_wwh signal_zzh_1FJ signal_wzh_zzh_2FJ
            python python/make_histos.py -c $channel -y $year -r $region
        done
    done
done

for channel in run3_1FJ ; do           # run3_1FJ run3_2FJ
    for year in 2024 ; do              # 2022, 2023 on disk but missing from common_mc
        for region in signal_wwh ; do  # signal_wwh signal_zzh_1FJ signal_wzh_zzh_2FJ
            python python/make_histos.py -c $channel -y $year -r $region
        done
    done
done

# —————————— Plot Histograms ————————————————————————————————————————
for channel in run2_1FJ ; do           # run2_1FJ run2_2FJ
    for year in 2016 ; do              # 2016APV 2016 2017 2018
        for region in signal_wwh ; do  # signal_wwh signal_zzh_1FJ signal_wzh_zzh_2FJ
            python python/plot_histos.py -c $channel -y $year -r $region
        done
    done
done

for channel in run3_1FJ ; do           # run3_1FJ run3_2FJ
    for year in 2024 ; do              # 2022, 2023 on disk but missing from common_mc
        for region in signal_wwh ; do  # signal_wwh signal_zzh_1FJ signal_wzh_zzh_2FJ
            python python/plot_histos.py -c $channel -y $year -r $region
        done
    done
done