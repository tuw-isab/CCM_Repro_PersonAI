import pickle
from pathlib import Path

pcm_dict = {}

for f in Path("Dorn_random_forest_pickles").glob("*.pickle"):
    user_id = f.name.split(".")[0]
    with open(f, "rb") as file:
        pcm_dict[user_id] = pickle.load(file)
        print(f"Loaded PCM for user_id: {user_id}")

with open("Dorn_random_forest_pickles/dorn_pcm_random_forest_all_users_models.pickle", "wb") as f:
    pickle.dump(pcm_dict, f)