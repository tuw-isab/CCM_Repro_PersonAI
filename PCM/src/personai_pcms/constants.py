from src_anchor import src_dir

LABELS = [9, 10, 11]
TARGET_COL = "thermal_cozie"
POINT_COUNT_PER_USER = 60
RANDOM_SEED = 42


# Person AI (active; the DORN settings below can be used instead by commenting this block and uncommenting one of them)
FEATURE_COLS = ["t_ubi", "rh_ubi", "hr_ubi", "bpm_time_diff_log", "CO2", "activity"] #hr_ubi instead of bpm
DATASET_PATH = src_dir() + "../../data/personai/final_main_dataset_no_missing_ALLENC_with_time_and_hr_age.csv"
RESULTS_PATH = src_dir() + f'../results/personai/'
FIGS_TABLES_PATH = src_dir() + "../figs_and_tables/personai/"
CATEGORICAL_FEATURES = []
MAKE_HOTD = False

# # Person AI with hotd
# FEATURE_COLS = ["t_ubi", "rh_ubi", "bpm", "bpm_time_diff_log", "CO2", "activity", "hotd"]
# DATASET_PATH = src_dir() + "../dataset/personai/final_main_dataset_no_missing_ALLENC_with_time_and_hr_age.csv"
# RESULTS_PATH = src_dir() + f'../results/personai/'
# FIGS_TABLES_PATH = src_dir() + "../figs_and_tables/personai/"
# CATEGORICAL_FEATURES = []
# MAKE_HOTD = True

# # DORN like in ccm paper
# FEATURE_COLS = ["clothing", "heartRate", "t-nb", "t-ubi", "rh-ubi"]
# DATASET_PATH = src_dir() + "../../data/dorn/DORN_env_phy_231.csv"
# RESULTS_PATH = src_dir() + f'../results/dorn/'
# FIGS_TABLES_PATH = src_dir() + "../figs_and_tables/dorn/"
# CATEGORICAL_FEATURES = ["clothing"]
# MAKE_HOTD = False

# # DORN but without clothing
# FEATURE_COLS = ["heartRate", "t-nb", "t-ubi", "rh-ubi"]
# DATASET_PATH = src_dir() + "../dataset/dorn/DORN_env_phy_231.csv"
# RESULTS_PATH = src_dir() + f'../results/dorn/'
# FIGS_TABLES_PATH = src_dir() + "../figs_and_tables/dorn/"
# CATEGORICAL_FEATURES = ["clothing"]
# MAKE_HOTD = False

