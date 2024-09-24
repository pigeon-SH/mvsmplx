import os
root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result"
logs = [
        "0922_default_sapiens",
        "0922_default_openpose",
        "0922_kptsmask_openpose",
        # "0922_kptsmask_sapiens",
        "0922_temporal_openpose",
        # "0922_temporal_sapiens",
        "0922_ranking_openpose",
        # "0922_ranking_sapiens",   
    ]

for log in logs:
    results = {}
    render_result_path = os.path.join(root, log, "evaluation.txt")
    with open(render_result_path, "r") as f:
        lines = f.readlines()
    results["CHAMFER"] = float(lines[0].split()[2])
    results["P2S"] = float(lines[0].split()[4])
    
    
    print(f"{log:25}", f"CHAMFER: {results['CHAMFER']:5.4f} P2S: {results['P2S']:5.4f}")