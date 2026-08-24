from __future__ import annotations

DATA_SAMPLES = ["JetMET", "Muon", "ParkingHH", "ParkingSingleMuon", "ParkingVBF"]

LUMI = {
    "2018": 59800,
    "2022": 7980.5,
    "2022EE": 26671.6,
    "2023": 18084.4,
    "2023BPix": 9692.1,
    "2024": 109080.0,
    "2022-2023": 62428.6,
    "2022-2024": 171508.6,
    "2018": 59830.0,
    "2017": 41480.0,
    "2016": 16810.0,
    "2016APV": 19520.0,
    "2016-2018": 137640.0
}

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

data_key = "data"
