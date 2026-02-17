from pathlib import Path

TEST_IMAGES = [
    Path("01/hela-20min-1_R3D.png"),
    Path("02/hela-20min-4_R3D.png"),
    Path("03/hela-20min-10_R3D.png"),
    Path("04/hela-30min-1_R3D.png"),
    Path("05/hela-30min-2_R3D.png"),
    Path("06/hela-30min-9_R3D.png"),
    Path("07/hela-40min-1_R3D.png"),
    Path("08/hela-40min-3_R3D.png"),
    Path("09/hela-40min-11_R3D.png"),
    Path("10/hela-60min-3_R3D.png"),
    Path("11/hela-60min-5_R3D.png"),
    Path("12/hela-60min-6_R3D.png"),
]

GTS = [f.parent / "Results" / "GroundTruth.png" for f in TEST_IMAGES]
AE_GTS = [f.parent / "AE_Segm.png" for f in TEST_IMAGES]
EF_GTS = [f.parent / "EF_Segm.png" for f in TEST_IMAGES]
MM_GTS = [f.parent / "MM_Segm.png" for f in TEST_IMAGES]
