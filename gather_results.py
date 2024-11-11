import os
root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result"
logs = [
        # "0922_default_sapiens",
        # "0922_default_openpose",
        # "0922_kptsmask_openpose",
        # "0922_kptsmask_sapiens",
        # "0922_temporal_openpose",
        # "0922_temporal_sapiens",
        # "0922_ranking_openpose",
        # "0922_ranking_sapiens",   
        # "0924_default_single_maskpred",
        # "0924_kptsmask_single_maskpred",
        # "0924_temporal_single_maskpred",
        # "0924_ranking_single_maskpred",
        # "0925_temporal_single_maskpred_nokptsmask",
        # "0925_ranking_single_maskpred_nokptsmask",
        "1017_temporal_7view",
        "1106_temporal_7view",
        "1106_temporal_7view_maskfilter"
    ]
seqs = [
    # "backhug/backhug02",
    "hug/hug01",
    "talk/talk22",
]

for seq in seqs:
    for log in logs:
        results = {}
        render_result_path = os.path.join(root, log, seq, "evaluation.txt")
        with open(render_result_path, "r") as f:
            lines = f.readlines()
        results["CHAMFER"] = float(lines[0].split()[2])
        results["P2S"] = float(lines[0].split()[4])
        
        
        print(f"{log:40}", f"CHAMFER: {results['CHAMFER']:5.4f} P2S: {results['P2S']:5.4f}")