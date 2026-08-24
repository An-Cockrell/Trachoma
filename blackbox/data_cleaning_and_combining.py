import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

socit = pd.read_csv("../data/SOCIT_key.csv")
# socit = socit.drop(columns='TF RE')
socit = socit[socit["TF LE"].notna()]
socit["TF"] = (socit["TF LE"] >= 1).astype(int)
socit["source"] = "SOCIT"
socit["dir_path"] = "/home/Trachoma/data/SOCIT_R/"
socit = socit[["PHOTO ID", "TF", "source", "dir_path"]]
socit_dir = Path("/home/Trachoma/data/SOCIT_R/")
# 1) collect filenames
socit_files = [
    p.name for p in socit_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
]
# 2) build file table: photoID = part before first "_"
df_socit_files = pd.DataFrame({"filename": socit_files})
df_socit_files["PHOTO ID"] = (
    df_socit_files["filename"].str.lstrip("0").str.split("_", n=1).str[0]
)
socit["PHOTO ID"] = socit["PHOTO ID"].astype(str)
df_socit_files["PHOTO ID"] = df_socit_files["PHOTO ID"].astype(str)
socit = socit.merge(df_socit_files, on="PHOTO ID", how="inner")
socit["filepath"] = socit["dir_path"] + socit["filename"]
socit.rename(columns={"PHOTO ID": "id"}, inplace=True)

cc_ea2017 = pd.read_csv("../data/CC_2017_keyV1.csv")  # use column "tf_final_gc_grade"
cc_ea2017 = cc_ea2017[cc_ea2017["tf_final_gc_grade"].notna()]
cc_ea2017["TF"] = cc_ea2017["tf_final_gc_grade"].astype(int)
cc_ea2017["source"] = "CC_EA2017"
cc_ea2017["dir_path"] = "/home/Trachoma/data/CC_EA2017/"
cc_ea2017 = cc_ea2017[
    ["eyeuid", "image", "person_unique_id", "TF", "source", "dir_path"]
]
cc_ea2017.rename(columns={"image": "filename"}, inplace=True)
cc_ea2017["filepath"] = cc_ea2017["dir_path"] + cc_ea2017["filename"]
cc_ea2017 = cc_ea2017.rename(columns={"eyeuid": "id"}).drop(columns="person_unique_id")

tana_and_solomons = pd.read_csv("../data/TF/metadata.csv")
tana_and_solomons = tana_and_solomons[
    tana_and_solomons.source.isin(
        [
            "2022 Australia Trachoma Images",
            "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011",
            "Gambia PRET 18m",
            "Solomon Islands research study 2015",
        ]
    )
]
# use column "TD consensus TF"
# tana_and_solomons = tana_and_solomons[['id', 'filename', 'source', 'Country', 'Number of original graders', 'TD consensus TF']]
tana_and_solomons = tana_and_solomons[tana_and_solomons["TD consensus TF"].notna()]
tana_and_solomons["TF"] = tana_and_solomons["TD consensus TF"].astype(int)
source_to_dir = {
    "2022 Australia Trachoma Images": "/home/Trachoma/data/TF/Australia/",
    "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011": "/home/Trachoma/data/TF/TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011/",
    "Gambia PRET 18m": "/home/Trachoma/data/TF/Gambia PRET 18m/",
    "Solomon Islands research study 2015": "/home/Trachoma/data/TF/Solomons2/",
}
tana_and_solomons["dir_path"] = tana_and_solomons.source.map(source_to_dir)
tana_and_solomons = tana_and_solomons[["id", "filename", "source", "TF", "dir_path"]]
tana_and_solomons["filepath"] = (
    tana_and_solomons["dir_path"] + tana_and_solomons["filename"]
)

tana_dup_key = pd.read_csv("/home/Trachoma/data/TF/TDkeyduplicates.csv")
tana_dup_key = tana_dup_key[~tana_dup_key.study.isin(["Tanzania", "Gambia"])]
replacement_dict = {
    "Australia": "2022 Australia Trachoma Images",
    "Ethiopia": "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011",
    "Solomons": "Solomon Islands research study 2015",
}
tana_dup_key["study"] = tana_dup_key["study"].replace(replacement_dict)
tana_dup_key = tana_dup_key.rename(columns={"study": "source"})

tana = tana_dup_key[
    tana_dup_key["source"]
    == "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011"
]
tana = tana.rename(columns={"participantid": "id"})
tana = tana[["source", "filename", "id"]]


australia = tana_dup_key[tana_dup_key["source"] == "2022 Australia Trachoma Images"]
australia["canonical"] = australia["filename"]
australia.loc[australia["duplicate"] > 1.0, "canonical"] = australia.loc[
    australia["duplicate"] > 1.0, "dupid"
]
# Step 2: assign unique IDs per canonical image
australia["id"] = pd.factorize(australia["canonical"])[0]
australia = australia[["source", "filename", "id"]]


solomons = tana_dup_key[tana_dup_key["source"] == "Solomon Islands research study 2015"]
solomons["canonical"] = solomons["filename"]
solomons.loc[solomons["duplicate"] > 1.0, "canonical"] = solomons.loc[
    solomons["duplicate"] > 1.0, "dupid"
]
# Step 2: assign unique IDs per canonical image
solomons["id"] = pd.factorize(solomons["canonical"])[0]
solomons = solomons[["source", "filename", "id"]]
tana_dup_key = pd.concat([tana, australia, solomons])

tana_and_solomons.drop(columns="id", inplace=True)
tana_and_solomons = pd.merge(
    tana_and_solomons, tana_dup_key, how="left", on=["source", "filename"]
)

# Gambia is intentionally excluded from the duplicate key (no canonical-duplicate
# info exists for it), so its id is NaN after the left-merge. Enumerate it per
# image — the same scheme used for the other sources without a built-in subject
# id (Kim, ICAPS). Without this, group_stratified_split drops NaN group keys and
# all Gambia images fall out of train/val/test entirely.
_gambia = tana_and_solomons["source"] == "Gambia PRET 18m"
tana_and_solomons.loc[_gambia, "id"] = np.arange(int(_gambia.sum()))


img_dir_o = "/home/Trachoma/TrachomaData/tarsal plate zip/allTZphotos/allTZphotos/"
img_keys_o = "/home/Trachoma/2300consensus8-2021.csv"
img_dir_m = "/home/Trachoma/m/"
img_keys_m = "/home/Trachoma/m/tfti.csv"

kim_et_al = pd.read_csv(img_keys_m)
kim_et_al.drop(columns=["TI"], inplace=True)
kim_et_al.rename(columns={"key": "id"}, inplace=True)
kim_et_al["source"] = "Kim et al"
kim_et_al["dir_path"] = img_dir_m
kim_et_al["filename"] = kim_et_al["id"].apply(lambda x: "image" + str(x) + ".jpg")
kim_et_al["filepath"] = kim_et_al["dir_path"] + kim_et_al["filename"]


icaps = pd.read_csv(img_keys_o)
icaps.rename(columns={"consensus": "TF"}, inplace=True)
icaps["source"] = "ICAPS"
icaps["dir_path"] = img_dir_o
icaps["filename"] = icaps["imagename"].apply(lambda x: "0" + str(x) + ".jpg")
icaps["filepath"] = icaps["dir_path"] + icaps["filename"]
icaps.drop(columns=["inputimage_url", "imagename"], inplace=True)
icaps["id"] = range(0, len(icaps))
all_metadata = pd.concat(
    [socit, cc_ea2017, tana_and_solomons, kim_et_al, icaps], ignore_index=True
)

all_metadata.rename(
    columns={"TF": "label", "filename": "image_name", "filepath": "image_path"},
    inplace=True,
)
all_metadata = all_metadata.reset_index(drop=True)
# filter the bad images out

# print("scanning for bad images...")
# bad = []
# for i, p in enumerate(
#     tqdm(all_metadata["image_path"], total=len(all_metadata), mininterval=1)
# ):
#     try:
#         with Image.open(p) as im:
#             im = im.convert("RGB")
#             im.load()  # <-- forces full decode, catches truncation
#     except Exception as e:
#         bad.append((i, p, repr(e)))

# print("Bad images:", len(bad))
# for i, p, e in bad[:20]:
#     print(i, p, e)

# bad_df = pd.DataFrame(bad, columns=["row_idx_in_all_metadata", "image_path", "error"])
# bad_df.to_csv("/home/Trachoma/data/corrupted_images.csv", index=False)
bad = pd.read_csv('/home/Trachoma/data/corrupted_images.csv')
# print(len(bad))
# print(len(all_metadata))
all_metadata_clean = all_metadata.drop(index=bad['row_idx_in_all_metadata'].to_list()).reset_index(drop=True)
all_metadata_clean.to_csv('/home/Trachoma/data/all_metadata.csv', index=False)
# print(len(all_metadata_clean))