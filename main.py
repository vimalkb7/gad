from src.data import generate_sbm_data
from src.experiments.sbm import run_sbm_training, run_sbm_experiments
from src.experiments.metrla import run_metrla_training, run_metrla_experiments
from src.experiments.molene import run_molene_training, run_molene_experiments
from src.plots.ammd import plot_ammd


TRAIN = True       # Set to False to reuse the denoisers stored in checkpoints/


if __name__ == "__main__":
    generate_sbm_data()
    if TRAIN:
        run_sbm_training()
    run_sbm_experiments()

    if TRAIN:
        run_metrla_training()
    run_metrla_experiments()

    if TRAIN:
        run_molene_training()
    run_molene_experiments()

    plot_ammd()
