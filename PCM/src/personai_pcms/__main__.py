from personai_pcms.evaluation import evaluate_pcms
from personai_pcms.pcm_model_training_and_cv import run_pcm_experiment

if __name__ == "__main__":
    run_pcm_experiment()
    evaluate_pcms()
