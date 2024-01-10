#!/usr/bin/env python
import argparse
import logging
import os
import sys
from dataclasses import asdict
import math
import datetime
import pickle
import shutil
import warnings
import time
import random
import pandas as pd
import osmnx as ox
import pathlib
from pathlib import Path
import tempfile
import ray
import optuna
from masters import *

from deap import creator, base, tools
from types import SimpleNamespace

from sys import platform

# You can change this code to decide if you want to perform calculations
# in series or in parallel. Generally, I recommend using series (osrm_batch_router)
# if you are creating one Ray worker per available CPU.
# Otherwise, if running just a single worker, use the parallel version.
# The code below is the configuration I used; feel free to try something else
# depending on the calculation infrastructure that you have.
if platform == "linux" or platform == "linux2":
    OSRM_BATCH_ROUTER_EXECUTABLE_NAME = "osrm_batch_router"
elif platform == "darwin": 
    OSRM_BATCH_ROUTER_EXECUTABLE_NAME = "osrm_batch_router_parallel"
elif platform == "win32":
    raise ValueError("""
                     Windows support has not been tested. It may work, but has never been tested.
                     Modify this code to remove this ValueError in order to try running on Windows.
                     """)

CODE_FOLDER_DIR = Path(__file__).parent.resolve()

logger = logging.getLogger('gen_algo')

def configure_logger(logger_, level):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(process)d - %(funcName)s: %(message)s'))
    logger_.addHandler(ch)
    logger_.setLevel(level)
    return logger_

worker_pool = None

def creator_setup():
    with warnings.catch_warnings():
        # Ignore "RuntimeWarning: A class named 'FitnessMin' has already been created and it will be overwritten."
        warnings.simplefilter("ignore", category=RuntimeWarning)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin, network_length=float, feasible=bool, history_index=int)

def create_toolbox(parameters):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, parameters.GENE_MIN, parameters.GENE_MAX)
    toolbox.register("mate", parameters.MATE_FUNC)
    toolbox.register("mutate", parameters.MUTATE_FUNC, indpb=parameters.MUTPB)
    toolbox.register("select", parameters.SELECT_FUNC, tournsize=parameters.TOURN_SIZE)
    return toolbox

#print(os.environ.get('ray_head_node_ip'))
ray.init(address=os.environ.get('ray_head_node_ip'))

#############################################################################
# Genetic algorithm worker
#############################################################################
from ray.util import ActorPool
@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class GeneticAlgorithmWorker():
    def __init__(self, edges_data=None, params=None, worker_id=None, master_directory=None, od_df_path=None, graph_path=None, creator_setup=None, loglevel=logging.DEBUG):
        # recreate scope from global
        if creator_setup is not None:
            creator_setup()

        # NOTE: If the calculations seem really slow, it might be because
        # of slow disk IO on network filesystems.
        # ** MAKE SURE that the temp folder is being created on a local file system (NOT networked!) **
        worker_directory = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())) / "genetic_algorithm" / f"worker_{worker_id}"
        if worker_directory.exists() and worker_directory.is_dir():
            shutil.rmtree(worker_directory) # Empty the directory if it exists
        worker_directory.mkdir(parents=True)

        # Copy master data to worker directory.
        #shutil.copytree(master_directory, worker_directory, dirs_exist_ok=True)
        prepare_optimization(directory=worker_directory,
            osm_xml_path=params.DATA_FOLDER / "osrm_network.xml",
            lua_profile_path=CODE_FOLDER_DIR / 'osrm_profiles' / 'bicycle_profile_with_csv_edge_filter.lua')

        self.logger = logging.getLogger(__name__)
        self.logger = configure_logger(self.logger, loglevel)

        self.edges_data = edges_data
        self.params = params
        self.worker_directory = worker_directory
        self.logger.debug("Hello from worker %d -- %s" % (worker_id, self.worker_directory))
        self.logger.debug("The next step might take a while (several minutes), please be patient...")

        od_df = pd.read_csv(od_df_path)
        self.od_df_exploded = create_exploded_od_df(od_df, edges_data['gdf_links'])
        self.graph = ox.load_graphml(filepath=graph_path)

        self.logger.debug("Worker is ready. %d -- %s" % (worker_id, self.worker_directory))

    def evaluate_individual(self, individual, parameters):
        edges_df = self.edges_data['gdf_links']

        return calculate_fitness(individual=individual, edges_df=edges_df,
                        working_directory=self.worker_directory, # The directory containing the 'osrm' folder as well as other relevant files (od_trips.csv, etc)
                        osrm_batch_router_path=CODE_FOLDER_DIR / "osrm_batch_router" / "build" / OSRM_BATCH_ROUTER_EXECUTABLE_NAME, # Full path to osrm_batch_router
                        od_df_exploded=self.od_df_exploded,
                        G=self.graph,
                        optimization_params=parameters)

#############################################################################
# Main code
#############################################################################

def save_run_to_global_logbook(trial, repetition, min_fit, logbook_path):
    df = pd.DataFrame([{'trial': trial, 'repetition': repetition, 'min_fit': min_fit}])
    with open(logbook_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False) # Only print header if file is empty

def run_genetic_algorithm(params,
                  toolbox,
                  edges_df,
                  simulations_folder,
                  checkpoint=None):
    start_timestamp = time.time()

    if checkpoint: # TODO: this hasn't been tested in a long time, but technically there is a way to resume a run after stopping it.
        # A file #name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"] + 1 # The starting generation is the last completed generation, plus one.
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        population = [creator.Individual(generate_individual(edges_df=edges_df, max_length=params.MAX_CYCLING_NETWORK_LENGTH)) for _ in range(params.NPOP)]#toolbox.population(n=params.NPOP)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=100)
        logbook = tools.Logbook()

        save_genealogy = hasattr(params, 'SAVE_GENEALOGY') and params.SAVE_GENEALOGY
        if save_genealogy:
            history = tools.History()
            #toolbox.decorate("mate", history.decorator)
            #toolbox.decorate("mutate", history.decorator)

        logger.debug("Starting eval of initial population")
        # Evaluate the fitness of the initial population.
        results_gen = worker_pool.map(lambda a, ind: a.evaluate_individual.remote(np.array(ind), params), population)
        for ind, result in zip(population, results_gen):
            ind.fitness.values = (result.fitness,)
            ind.result_data = result
            ind.network_length = result.network_length
            ind.feasible = result.network_length <= params.MAX_CYCLING_NETWORK_LENGTH
            ind.trips = result.trips
        
        if save_genealogy:
            history.update(population)
        
        halloffame.update(population)

        logger.debug("Finished eval of initial population")

    # Save the simulation parameters in the simulations folder
    with open(simulations_folder / "simulation_params.json", 'w', encoding='utf-8') as f:
        params_sanitized = {}
        for k, v in params.__dict__.items():
            if type(v) == types.FunctionType or isinstance(v, pathlib.PurePath) == True:
                params_sanitized[k] = str(v)
            else:
                params_sanitized[k] = v
    
        json.dump(params_sanitized, f, ensure_ascii=False, indent=4)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("fit_avg", np.mean)
    stats.register("fit_std", np.std)
    stats.register("fit_min", np.min)
    stats.register("fit_max", np.max)

    msg = "Starting genetic algorithm... trial = (%d, %d)" % (params.TRIAL, params.REPETITION) if hasattr(params, 'TRIAL') else "Starting genetic algorithm..."
    #msg = "Starting genetic algorithm..."
    logger.debug(msg)
    print(params.MATE_FUNC, toolbox.mate, params.MUTATE_FUNC, toolbox.mutate)

    mutpb = params.MUTPB

    best_inds_log = dict()
    best_inds_log[0] = halloffame[0]

    for gen in range(start_gen, params.NGEN+1):
        gen_start_timestamp = time.time()

        next_population = []
        # Elitism: automatically include the top N_ELITES individuals
        # from the previous generation in the next generation.
        population = sorted(population, key=lambda ind: ind.fitness.values[0])
        elites = [toolbox.clone(ind) for ind in population[0:params.N_ELITES]]
        next_population.extend(elites)

        while len(next_population) < params.NPOP: # If true, we need to generate new offsprings
            # Shuffle population. (Does it matter since we do tournament selection right after?)
            random.shuffle(population)

            offspring = []

            # Step 1: Crossover
            if random.random() <= params.CXPB:
                # For cxOnePoint, we just need two parents. For probabilisticGeneCrossover, we need 4 parents.
                if params.MATE in ["cxOnePoint", "cxTwoPoint"]:
                    parents = [toolbox.clone(parent) for parent in toolbox.select(population, 2)]
                    offspring = list(toolbox.mate(parents[0], parents[1]))
                    del offspring[0].fitness.values
                    del offspring[1].fitness.values
                else:
                    for _ in range(2):
                        parents = [toolbox.clone(parent) for parent in toolbox.select(population, 2)]
                        if params.MATE == "probabilisticGeneCrossover":
                            child = toolbox.mate(parents[0], parents[1])
                        elif params.MATE == "tripsBasedCrossover":
                            child = toolbox.mate(parents[0], parents[1], edges_df=edges_df, params=params)
                        del child.fitness.values
                        offspring.append(child)
            else:
                # If we don't do crossover, just pick two parents
                parents = [toolbox.clone(parent) for parent in toolbox.select(population, 2)]

            # Step 2: Mutation
            for i in range(len(offspring)):
                if not math.isclose(mutpb, 0.0): # Only do mutation if the mutation rate is not zero.
                    offspring[i], = toolbox.mutate(offspring[i], indpb=mutpb)
                    del offspring[i].fitness.values

            next_population.extend(offspring)

        # Collect some interesting indicators
        total_durations         = []
        unreachable_trips_count = []
        network_lengths         = []

        # Step 3: Evaluate the individuals with an invalid fitness (they have mated or mutated)
        n_evals = 0
        population_to_eval = [ind for ind in next_population if not ind.fitness.valid]

        results_gen = worker_pool.map(lambda a, ind: a.evaluate_individual.remote(np.array(ind), params), population_to_eval)
        for ind, result in zip(population_to_eval, results_gen):
            ind.fitness.values = (result.fitness,)
            ind.result_data = result
            ind.network_length = result.network_length
            ind.feasible = result.network_length <= params.MAX_CYCLING_NETWORK_LENGTH
            ind.trips = result.trips

            n_evals += 1
            network_lengths.append(result.network_length)
            total_durations.append(result.total_duration)
            unreachable_trips_count.append(result.unreachable_trips)

        population = toolbox.clone(next_population)
        if save_genealogy:
            history.update(population)
        
        feasible_individuals = [ind for ind in population if ind.feasible]
        n_feasible = len(feasible_individuals)

        # Update the hall of fame, stats and logbook.
        halloffame.update(feasible_individuals)
        best_inds_log[gen] = halloffame[0]

        with warnings.catch_warnings():
            # Ignore numpy's warnings when calculating the mean of an empty array
            warnings.simplefilter("ignore", category=RuntimeWarning)
            record = stats.compile(population)# if n_feasible > 0 else {"fit_avg": np.nan, "fit_min": np.nan, "fit_max": np.nan, "fit_std": np.nan} # FIXME: kind of a hack...
            gen_duration = time.time() - gen_start_timestamp

            logbook.record(gen=gen,
                            mutpb=mutpb,
                            evals=n_evals,
                            t_per_eval=f"{gen_duration/max(n_evals, 1):0.2f}",
                            gen_dur=str(datetime.timedelta(seconds=gen_duration)).split(".")[0],
                            tot_runtime=str(datetime.timedelta(seconds=time.time() - start_timestamp)).split(".")[0],
                            net_len_mean=np.mean(network_lengths),
                            ratio_feas=f"{(n_feasible/len(population))*100:0.2f} %",
                            td_mean=np.mean(total_durations) if len(total_durations) > 0 else np.nan,
                            td_min=np.min(total_durations) if len(total_durations) > 0 else np.nan,
                            ut_mean=np.mean(unreachable_trips_count) if len(unreachable_trips_count) > 0 else np.nan,
                            ut_min=np.min(unreachable_trips_count) if len(unreachable_trips_count) > 0 else np.nan,
                            **record)

        logger.debug(logbook.stream)

        stop = False

        logbook_df = pd.DataFrame(logbook)
        if len(logbook_df) > params.PATIENCE:
            current_fit = logbook_df.fit_min.iloc[-1]
            # We should stop if it's been more than params.PATIENCE generations
            # since there was any significant improvement (i.e. params.MIN_DELTA %) in the best fitness obtained.
            best_fit_patience_ago = logbook_df.fit_min.iloc[-params.PATIENCE]
            if (best_fit_patience_ago - current_fit)/best_fit_patience_ago < params.MIN_DELTA:
                stop = True

        tot_runtime = time.time() - start_timestamp
        if tot_runtime > params.RUNTIME_LIMIT: # Limit runtime
            logger.debug("Stopping due to runtime limit: %d >= %d" % (tot_runtime, params.RUNTIME_LIMIT))
            stop = True

        if gen % params.CHECKPOINT_FREQ == 0 or stop: # Save a checkpoint when we stop
            #print("Saving gen:", gen)
            # Write a checkpoint containing useful information for restarting the simulation later
            cp = dict(params=params.__dict__,
                        #population=population,
                        generation=gen,
                        halloffame=halloffame,
                        best_inds_log=best_inds_log,
                        logbook=logbook,
                        rndstate=random.getstate())
        
            if save_genealogy:
                cp['history'] = history
            
            with open(simulations_folder / f"deap_checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

        step_size = 10
        if params.MUTATION_RATE_TYPE == "exponential_decay":
            mutpb = mutpb * params.MUTPB_DECAY_RATE ** (gen / step_size)
        elif params.MUTATION_RATE_TYPE == "step_decrease":
            if gen % step_size == 0 and gen > 0:
                mutpb = mutpb * params.MUTPB_STEP_DECREASE_GAMMA

        # Save logs
        logbook_df.to_csv(simulations_folder / "logbook.csv", float_format='%.5f', index=False)

        if stop:
            break

    min_fit = logbook_df.fit_min.min()
    logger.info("Stopping at generation %d with min fitness %f" % (gen, min_fit))

    # Create a file to indicate that this run is finished.
    (simulations_folder / ".finished").touch()

    #print(f"cost={logs.fitness_min.min()}; runtime={time.time() - start_timestamp}; status=SUCCESS; additional_info=test (single-objective)")
    # cost=0.5; runtime=0.01; status=SUCCESS; additional_info=test (single-objective)
    #print("DONE!")
    return min_fit

mate_funcs = {'probabilisticGeneCrossover': probabilistic_gene_crossover,
    'tripsBasedCrossover': trips_based_crossover,
    'cxOnePoint': tools.cxOnePoint,
    'cxTwoPoint': tools.cxTwoPoint}

def create_argparser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Genetic algorithm')
    parser.add_argument('-a', 
                        dest="auto", 
                        action=argparse.BooleanOptionalAction, 
                        help="Perform hyperparameter optimization using Optuna")
    parser.add_argument('--optuna_log', 
                        type=Path, 
                        help="Path to optuna log file. File name must end in .log")
    parser.add_argument('--random_search', 
                        action=argparse.BooleanOptionalAction,
                        help="Generate --n_pop individuals randomly and measure their fitness")
    parser.add_argument('--random_search_output', 
                        type=str, help="File path where to save report of random search")
    parser.add_argument('-d', '--debug',
                        help="Print a lot of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG,
                        default=logging.INFO)
    parser.add_argument('--data_folder', 
                        type=Path, 
                        required=True, 
                        help="The folder containing the input data created during the data preparation step.")
    parser.add_argument('--value_of_time', 
                        type=float, 
                        default=10, 
                        help="Value of time in model [default: 10 $/h]")
    parser.add_argument('--unreachable_trip_cost', 
                        type=float, 
                        default=30, 
                        help="Value of unreachable trip in model [default: 30 $]")
    parser.add_argument('--output_folder', 
                        type=Path, 
                        help="Folder where to output optimization results")
    parser.add_argument('--n_workers', 
                        type=int, 
                        default=2, 
                        help="Number of workers to use for parallel network quality evaluation. Use number of CPUs available on system [default: 2]")
    parser.add_argument('--n_pop', 
                        type=int, 
                        default=700, 
                        help="Size of population")
    parser.add_argument('--n_gen', 
                        type=int, 
                        default=1000, 
                        help="Number of genetic algorithms generations to run for. Early stopping possible with --runtime_limit, or combination of --max_gen_no_improvement and --min_delta_improvement [default: 1000]")
    parser.add_argument('--n_runs', 
                        type=int, 
                        default=1, 
                        help="Number of optimization runs of the to perform [default: 1]") # how many runs of the genetic algorithm to do
    parser.add_argument('--checkpoint_freq', 
                        type=int, 
                        default=1, 
                        help="Save a checkpoint after X numbers of generations [default: 1]. Will overrite previous checkpoint.")
    parser.add_argument('--runtime_limit', 
                        type=int,
                        default=3*60*60, # 3 hours in seconds
                        help="Max runtime (in seconds) for each genetic algorithm execution [default: 3 hours]")
    parser.add_argument('--save_genealogy', 
                        action='store_true',
                        help="Save genealogy tree during genetic algorithm run. Hasn't really been tested properly though.")
    parser.add_argument('--walking_speed',
                        type=float, 
                        default=5, 
                        help="Walking speed in km/h [default: 5 km/h]")
    parser.add_argument('--cxpb', 
                        type=float,
                        default=0.569, 
                        help="Crossover probability [default: 0.569]")
    parser.add_argument('--mutpb', 
                        type=float, 
                        default=0.092, 
                        help="Mutation probability [default: 0.092]")
    parser.add_argument('--n_elites_prop', 
                        type=float, 
                        default=0.2, 
                        help="Proportion of elites probability [default: 0.2]")
    parser.add_argument('--tourn_size', 
                        type=int, 
                        default=20, 
                        help="Size of tournament for individuals selection [default: 20]")
    parser.add_argument('--max_cycling_length_prop', 
                        type=float, 
                        default=0.25, 
                        help="Maximum size of cycling network (for budgetary constraint) as a proportion of the length of all edges in input network [default: 0.25]")
    parser.add_argument('--constraint_penalty', 
                        type=int, 
                        default=500000,
                        help="Value to add to fitness of individuals which do not respect the budgetary constraint [default: 500000]")
    parser.add_argument('--max_gen_no_improvement', 
                        type=int, 
                        default=30,
                        help="Terminate optimization after X generations with negligable improvement (see also --min_delta_improvement) [default: 30]")
    parser.add_argument('--min_delta_improvement',
                        type=float,
                        default=0.5/100,
                        help="What to consider to be a meaningful improvement in fitness, in percentage (see --max_gen_no_improvement) [default: 0.005]")
    parser.add_argument('--mate',
                        type=str,
                        choices=list(mate_funcs.keys()),
                        default='probabilisticGeneCrossover',
                        help="Crossover method [default: probabilisticGeneCrossover]")
    parser.add_argument('--mutation_rate_type',
                        type=str,
                        choices=["constant", "exponential_decay", "step_decrease"],
                        default='step_decrease',
                        help="Mutation decrease method [default: step_decrease]")
    parser.add_argument('--mutpb_decay_rate', 
                        type=float, 
                        default=0.95,
                        help="Mutation decay rate for exponential_decay method [default: 0.95]")
    parser.add_argument('--mutpb_step_decrease_gamma', 
                        type=float, 
                        default=0.858,
                        help="Mutation step reduction for step_decrease method [default: 0.858]")

    return parser

base_params = SimpleNamespace(
    GENE_MIN=0, # Binary genes (0 and 1)
    GENE_MAX=1,
    MUTATE_FUNC=tools.mutFlipBit, # Mutation operator
    SELECT_FUNC=tools.selTournament, # Selection operator
)

def merge_parameters(first, second):
    return SimpleNamespace(**first.__dict__, **second.__dict__)

def _optuna_run(trial, parameters, data, results_directory):
    npop = trial.suggest_int("npop", 20, 1000, step=20)
    cxpb = trial.suggest_float("cxpb", 0.5, 1.0)
    mutpb = trial.suggest_float("mutpb", 0.0001, 0.15)
    n_elites_prop = trial.suggest_float("n_elites_prop", 0.0, 0.30, step=0.01)
    tourn_size = trial.suggest_int("tourn_size", 2, 50, step=2)
    #mate = trial.suggest_categorical("mate", ["probabilisticGeneCrossover", "tripsBasedCrossover", "cxOnePoint", "cxTwoPoint"])
    mate = trial.suggest_categorical("mate", ["probabilisticGeneCrossover", "cxOnePoint", "cxTwoPoint"])
    constraint_penalty = trial.suggest_int("constraint_penalty", 500000, 1000000, step=100000)

    #for previous_trial in trial.study.trials:
    #    if previous_trial.state == optuna.trial.TrialState.COMPLETE and trial.params == previous_trial.params:
    #        logging.debug("Duplicated trial: %s, returning previous value %f from trial %d" % (repr(trial), trial.number, previous_trial.value))
    #        return previous_trial.value

    mutation_rate_type = trial.suggest_categorical("mutation_rate_type", ["constant", "exponential_decay", "step_decrease"])
    new_parameters = SimpleNamespace(
        NPOP=npop, # Number of individuals in each generation
        CXPB=cxpb, # Probability of crossover
        MUTPB=mutpb, # Probability of mutating an individual
        MUTATION_RATE_TYPE=mutation_rate_type,
        N_ELITES=int(n_elites_prop * npop), # Number of elites
        TOURN_SIZE=tourn_size,  # Tournament size
        CONSTRAINT_PENALTY=constraint_penalty,
        MATE=mate,
        MATE_FUNC=mate_funcs[mate], # Crossover operator
    )

    parameters = merge_parameters(parameters, new_parameters)

    if mutation_rate_type == 'exponential_decay': # new_mutpb = initial_mutpb * decay_rate ^ (gen / gen_steps)
        decay_rate = trial.suggest_float("mutpb_decay_rate", 0.90, 0.99)
        parameters = merge_parameters(parameters, SimpleNamespace(MUTPB_DECAY_RATE=decay_rate))
    elif mutation_rate_type == 'step_decrease':
        gamma = trial.suggest_float("mutpb_step_decrease_gamma", 0.70, 0.99)
        parameters = merge_parameters(parameters, SimpleNamespace(MUTPB_STEP_DECREASE_GAMMA=gamma))

    gdf_links = data['gdf_links']

    logger.debug("_optuna_run: trial #%d, parameters: %s" % (trial.number, repr(parameters)))

    toolbox = create_toolbox(parameters)

    start_n = 0
    scores = []
    logbook_path = results_directory / "trial_logbook.csv"
    try:
        logbook = pd.read_csv(logbook_path)
        start_n = len(logbook[logbook.trial == trial.number].repetition) # Find how many repetitions have been done so far
        scores = logbook[logbook.trial == trial.number].min_fit.to_list() # Get the previous min fitnesses obtained
    except FileNotFoundError:
        pass # If the file does not exist it means that this is likely the first Optuna trial.

    if start_n > 0:
        logger.debug("Restarting at n=%d for trial #%d. Fitnesses: %s" % (start_n, trial.number, scores))

    for repetition in range(start_n, parameters.N_REPETITIONS):
        parameters_ = merge_parameters(parameters, SimpleNamespace(TRIAL=trial.number, REPETITION=repetition))

        # Create a folder to store the results of this run
        simulations_folder = results_directory / f"trial_{parameters_.TRIAL}-run_{parameters_.REPETITION}"
        simulations_folder.mkdir(parents=True, exist_ok=True)

        score = run_genetic_algorithm(toolbox=toolbox, params=parameters_, edges_df=gdf_links, simulations_folder=simulations_folder)
        scores.append(score)

        save_run_to_global_logbook(trial=parameters_.TRIAL,
            repetition=repetition,
            min_fit=score,
            logbook_path=results_directory / "trial_logbook.csv")

        # Report intermediate objective value.
        trial.report(score, step=repetition)

        # Check if we should prune this trial.
        # We allow pruning only if:
        #   1) we have done enough repetitions for this trial
        #   2) we have completed enough trials to have an idea of the global best fitness landscape to compare it with.
        if len(scores) >= parameters_.PRUNING_MIN_REPETITIONS and trial.number >= parameters_.PRUNING_WARMUP_TRIALS:
            # We should prune if the mean of these repetitions is worse than the mean of the previous trials.
            previous_trials = logbook[logbook.trial < trial.number] # Get the trials before this one
            previous_trials_median = previous_trials["min_fit"].median()
            if np.median(scores) > previous_trials_median:
                logger.debug("Pruning trial %d at repetition %d since %f > %f" % (trial.number, repetition, np.median(scores), previous_trials_median))
                raise optuna.TrialPruned()
            else:
                logger.debug("Not pruning for trial (%d, %d) since np.median(scores)=%f <= previous_trials_median=%f" % (trial.number, repetition, np.median(scores), previous_trials_median))
        else:
            logger.debug("Not allowing pruning for (%d, %d) since %d <= %d or %d <= %d" % (trial.number, repetition, len(scores), parameters_.PRUNING_MIN_REPETITIONS, trial.number, parameters_.PRUNING_WARMUP_TRIALS))

    mean_best_fitness = sum(scores) / len(scores)
    logger.debug("Finished trial after %d repetitions. Obtained mean best fitness: %f" % (len(scores), mean_best_fitness))

    # Return average best fitness obtained
    return mean_best_fitness

def _setup_simulation(parameters):
    global worker_pool
    logger.debug("Loading gdf_links...")
    # TODO: clean up this mess of repeated variables here and in GeneticAlgorithmWorker.
    graph_path = parameters.DATA_FOLDER / "G_simplified.graphml"
    gdf_links = ox.graph_to_gdfs(ox.load_graphml(filepath=graph_path), nodes=False)
    data = {'gdf_links': gdf_links}

    parameters = merge_parameters(parameters, SimpleNamespace(
        TOTAL_AVAILABLE_CYCLING_NETWORK_LENGTH=gdf_links['length'].sum(),
        MAX_CYCLING_NETWORK_LENGTH=parameters.MAX_CYCLING_LENGTH_PROP * gdf_links['length'].sum()
    ))

    # DEAP configuration
    creator_setup()

    scratch_dir = Path(tempfile.gettempdir()) / os.environ['USER'] / "genetic_algorithm"
    master_directory = scratch_dir / "master"
    if scratch_dir.exists() and scratch_dir.is_dir():
        shutil.rmtree(scratch_dir) # Empty the directory
    scratch_dir.mkdir(parents=True) # (Re-)create the directory

    logger.debug('Preparing %d workers' % parameters.N_WORKERS)

    workers = []

    for worker_id in range(1, parameters.N_WORKERS+1):
        workers.append(GeneticAlgorithmWorker.remote(creator_setup=creator_setup,
            params=parameters,
            edges_data=data,
            od_df_path=parameters.DATA_FOLDER / "od_df_filtered.csv",
            graph_path=graph_path,
            worker_id=worker_id,
            master_directory=master_directory
        ))

    worker_pool = ActorPool(workers)

    return parameters, data

def run_optuna(args):
    parameters = SimpleNamespace(
        N_WORKERS=args.n_workers,
        RUNTIME_LIMIT=args.runtime_limit,
        WALKING_SPEED=args.walking_speed,
        PATIENCE=args.max_gen_no_improvement,
        MIN_DELTA=args.min_delta_improvement,
        MAX_CYCLING_LENGTH_PROP=args.max_cycling_length_prop,
        DATA_FOLDER=args.data_folder,
        OPTUNA_LOG_PATH=args.optuna_log,
        NGEN=5000, # Number of generations; arbitrarily large number of generations to let it run until a combination of min_delta_improvement and max_gen_no_improvement
        CHECKPOINT_FREQ=5,
        N_REPETITIONS=10, # Number of repetitions of each trial
        PRUNING_MIN_REPETITIONS=5, # Do at least X steps in a trial before pruning
        PRUNING_WARMUP_TRIALS=40, # Do at least X trials before pruning
        VALUE_OF_TIME=args.value_of_time, # $/h
        UNREACHABLE_TRIP_COST=args.unreachable_trip_cost, # $
    )
    parameters = merge_parameters(parameters, base_params)
    parameters, data = _setup_simulation(parameters)

    study_name = str(parameters.OPTUNA_LOG_PATH)  # Unique identifier of the study.
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(parameters.OPTUNA_LOG_PATH)),
    )
    logger.debug("optuna: created storage")

    study = optuna.create_study(study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=parameters.PRUNING_WARMUP_TRIALS),
    )
    logger.debug("loaded study with sampler %s!" % (str(study.sampler.__class__.__name__)))

    logger.debug("Starting Optuna...")
    results_directory = args.output_folder / f"{study_name}-results" 
    study.optimize(lambda trial: _optuna_run(trial, parameters, data, results_directory=results_directory), n_trials=5000)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best = study.best_trial
    print("  Value: ", best.value)
    print("  Params: ")
    for key, value in best.params.items():
        print("    {}: {}".format(key, value))
    print("  User attrs:")
    for key, value in best.user_attrs.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    logger = configure_logger(logger, args.loglevel)
    logger.debug("Entry point")

    if args.auto: # Optimize hyperparameters with Optuna
        run_optuna(args)
    else:
        parameters = SimpleNamespace(
            N_WORKERS=args.n_workers,
            RUNTIME_LIMIT=args.runtime_limit,
            WALKING_SPEED=args.walking_speed,
            DATA_FOLDER=args.data_folder,
            NGEN=args.n_gen, # Number of generations
            NPOP=args.n_pop, # Number of individuals in each generation
            CHECKPOINT_FREQ=args.checkpoint_freq, # Save a checkpoint every X generations
            MAX_CYCLING_LENGTH_PROP=args.max_cycling_length_prop,
            SAVE_GENEALOGY=args.save_genealogy, # Save genetic algorithm genealogy tree to pickle file
            CXPB=args.cxpb, # Probability of crossover
            MUTPB=args.mutpb, # Probability of mutating an individual
            MUTPB_DECAY_RATE=args.mutpb_decay_rate,
            MUTPB_STEP_DECREASE_GAMMA=args.mutpb_step_decrease_gamma,
            MUTATION_RATE_TYPE=args.mutation_rate_type,
            N_ELITES=int(args.n_elites_prop * args.n_pop), # Number of elites
            TOURN_SIZE=args.tourn_size,  # Tournament size
            CONSTRAINT_PENALTY=args.constraint_penalty,
            PATIENCE=args.max_gen_no_improvement,
            MIN_DELTA=args.min_delta_improvement,
            MATE=args.mate,
            MATE_FUNC=mate_funcs[args.mate], # Crossover operator
            VALUE_OF_TIME=args.value_of_time, # $/h
            UNREACHABLE_TRIP_COST=args.unreachable_trip_cost, # $
        )

        parameters = merge_parameters(parameters, base_params)

        print(parameters)
        logger.debug(str(parameters))
        parameters, data = _setup_simulation(parameters)
        toolbox = create_toolbox(parameters)

        gdf_links = data['gdf_links']

        if args.random_search:
            logger.info("Starting random search with %d candidate solutions" % args.n_pop)
            # Generator individuals randomely
            population = [generate_individual(edges_df=gdf_links, max_length=parameters.MAX_CYCLING_NETWORK_LENGTH) for _ in range(args.n_pop)]
            results_gen = worker_pool.map(lambda a, ind: a.evaluate_individual.remote(np.array(ind), parameters), population)
            results = []
            for ind, result in zip(population, results_gen):
                simulation_res = asdict(result)
                simulation_res['max_length'] = parameters.MAX_CYCLING_NETWORK_LENGTH
                simulation_res['individual'] = ind
                results.append(simulation_res)
            logger.info("Done! Writing to file %s" % args.random_search_output)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('fitness')
            results_df.to_csv(args.random_search_output, index=False)
        else: # Manual genetic algorithm mode
            root_simulations_folder = args.output_folder
            root_simulations_folder.mkdir(parents=True, exist_ok=True)
            for i in range(args.n_runs):
                # Create a folder to store the results of this run
                sim_folder = root_simulations_folder / f'run_{i}'
                if sim_folder.exists():
                    logger.info("Directory %s exists. Skipping run %d" % (sim_folder, i))
                    continue
                sim_folder.mkdir(parents=True, exist_ok=True)

                score = run_genetic_algorithm(toolbox=toolbox,
                                            params=parameters,
                                            edges_df=gdf_links,
                                            simulations_folder=sim_folder)
                logger.info("Run %d: best fitness obtained %f" % (i, score))
            logger.info("Nothing left to do. Exiting.")
