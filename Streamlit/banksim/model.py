import numpy as np
import simpy
import pandas as pd
import itertools

# default resources
N_OPERATORS = 2
N_LONG_OPERATORS = 2 #Long transactions such as loan applications and financial consultation


# default service time parameters (triangular)
#This is for short transactions
CALL_LOW = 5.0
CALL_MODE = 7.0
CALL_HIGH = 10.0

#This is for long transactions
LONG_CALL_LOW = 60
LONG_CALL_MODE = 70
LONG_CALL_HIGH = 100

#for modelling probabilistic behavior
LOW_UNIF = 0
HIGH_UNIF = 1

#Queueing behavior parameters
#for get_dynamic_mean_iat(current_time, PEAK_MIN, PEAK_MIN_STANDARD_DEVIATION, LONG_IAT, SHORT_IAT):
PEAK_MIN= 4*60 #minute at which peak time is, measured from opening hour (e.g. 180 mins from opening hours)
PEAK_MIN_STANDARD_DEVIATION= 3*60 #smaller PEAK_MIN_STANDARD_DEVIATION = much much higher traffic during peak hr
MEAN_IAT = 4 # default mean inter-arrival time (exp)
LONG_IAT = 5
SHORT_IAT= 4
#percent of long transactions that consist of all transactions processed by the branch

LONG_TRANSACT_MEAN_IAT = 75
LONG_TRANSACT_LONG_IAT = 180
LONG_TRANSACT_SHORT_IAT = 45

#simulate queueing behavior inside branch
#Suppose branches have a particular capacity for a particular queue

AREA_BRANCH = 150 #in sq. m  #should be 
AREA_DESK_OFFICE_RATIO = 0.95 #Suppose 95% of the area of the branch is for desks, offices, etc
AREA_DESK_OFFICE = AREA_DESK_OFFICE_RATIO*AREA_BRANCH
AREA_WAIT_RATIO = 1-AREA_DESK_OFFICE_RATIO #this should be a 0-> 1 slider in streamlit
AREA_WAIT = AREA_WAIT_RATIO*AREA_BRANCH
CUSTOMER_PREFERRED_AREA = 1 #suppose a customer likes having 1 sq. m of space while waiting
#in units of sq. m per person
#Make this adjustable also. The default should be estimated


CUSTOMER_CAPACITY = int(np.round(AREA_WAIT/CUSTOMER_PREFERRED_AREA)) #calculated customer capacity based on available waiting area and 
#preferred personal space of customer

# sampling settings
N_STREAMS = 4 #DO NOT change
DEFAULT_RND_SET = 0

# Boolean switch to simulation results as the model runs
TRACE = False

# run variables
RESULTS_COLLECTION_PERIOD = 60*8
N_METRICS = 4 #we have mean_wait_time, teller_util, outside_mean_wait_time, long_teller_util

class Triangular:
    def __init__(self, low, mode, high, random_seed = None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.mode = mode
        self.high = high
    
    def sample(self, size=None):
        return self.rand.triangular(self.low, self.mode, self.high, size=size)

class Exponential:
    def __init__(self, default_mean, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.default_mean = default_mean

    def sample(self, size=None, mean=None):
        """
        Draws a sample from an exponential distribution.
        
        Parameters:
        - size: int or tuple, optional — output shape
        - mean: float, optional — overrides default_mean for this sample

        Returns:
        - float or ndarray
        """
        use_mean = mean if mean is not None else self.default_mean
        return self.rand.exponential(use_mean, size=size)

class NormalDistribution:
    def __init__(self, mean, std_dev, random_seed = None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, size=None):
        return self.rand.normal(loc=self.mean, scale=self.std_dev, size=size)
    

    
class Experiment:
    """
    Encapsulates the concept of an experiment 🧪 with the urgent care
    call centre simulation model.

    An Experiment:
    1. Contains a list of parameters that can be left as defaults or varied
    2. Provides a place for the experimentor to record results of a run 
    3. Controls the set & streams of psuedo random numbers used in a run.
    
    """

    def __init__(
        self,
        random_number_set=DEFAULT_RND_SET,
        n_operators=N_OPERATORS,
        n_long_operators=N_LONG_OPERATORS,
        mean_iat=MEAN_IAT,
        long_call_low = LONG_CALL_LOW,
        long_call_mode = LONG_CALL_MODE,
        long_call_high=LONG_CALL_HIGH,
        call_low=CALL_LOW,
        call_mode=CALL_MODE,
        call_high=CALL_HIGH,
        short_iat = SHORT_IAT,
        long_iat = LONG_IAT,
        peak_min = PEAK_MIN,
        long_transact_mean_iat = LONG_TRANSACT_MEAN_IAT,
        long_transact_short_iat =  LONG_TRANSACT_SHORT_IAT,
        long_transact_long_iat = LONG_TRANSACT_LONG_IAT,
        peak_min_standard_deviation = PEAK_MIN_STANDARD_DEVIATION,
        n_streams=N_STREAMS,
        customer_capacity = CUSTOMER_CAPACITY,
        low_unif = LOW_UNIF,
        high_unif=HIGH_UNIF
    ):
        #get_dynamic_mean_iat(current_time, PEAK_MIN, PEAK_MIN_STANDARD_DEVIATION, LONG_IAT, SHORT_IAT):

        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams
        
        # store parameters for the run of the model
        self.n_operators = n_operators
        self.mean_iat = mean_iat
        self.n_long_operators = n_long_operators

        self.call_low = call_low
        self.call_mode = call_mode
        self.call_high = call_high
        self.long_call_low =long_call_low
        self.long_call_mode = long_call_mode
        self.long_call_high = long_call_high

        self.long_transact_long_iat = long_transact_long_iat
        self.long_transact_short_iat = long_transact_short_iat
        self.long_transact_mean_iat = long_transact_mean_iat

        self.low_unif = low_unif
        self.high_unif = high_unif

        self.customer_capacity = customer_capacity
        self.short_iat = short_iat
        self.long_iat = long_iat

        self.peak_min = peak_min
        self.peak_min_standard_deviation = peak_min_standard_deviation
        
        # resources: we must init resources after an Environment is created.
        # But we will store a placeholder for transparency
        self.operators = None
        self.cell = None
        self.long_operators = None

        # initialise results to zero
        self.init_results_variables()

        # initialise sampling objects
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling
        Parameters:
        ----------
        random_number_set: int
            Used to control the set of pseudo random numbers used by 
            the distributions in the simulation.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Create the distributions used by the model and initialise
        the random seeds of each.
        """
        # produce n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.random_number_set)
        self.seeds = seed_sequence.spawn(self.n_streams)

        # create distributions

        # call inter-arrival times
        self.arrival_dist = Exponential(
            self.mean_iat, random_seed=self.seeds[0]
        )

        # duration of call triage
        self.call_dist = Triangular(
            self.call_low,
            self.call_mode,
            self.call_high,
            random_seed=self.seeds[1],
        )

        self.long_call_dist = Triangular(
            self.long_call_low,
            self.long_call_mode, 
            self.long_call_high,
            random_seed=self.seeds[2]
        )


    def init_results_variables(self):
        """
        Initialise all of the experiment variables used in results
        collection.  This method is called at the start of each run
        of the model
        """
        # variable used to store results of experiment
        self.results = {}
        self.results["waiting_times"] = []
        self.results['outside_waiting_times'] = []

        # total operator usage time for utilisation calculation.
        self.results["total_short_transact_duration"] = 0.0
        self.results['total_long_transact_duration'] = 0.0

def service(identifier, env, args, customer_type):
    '''simulates service process for a teller
    1. queue and wait for teller 
    2. transaction
    3. exit system
    
    Params:
    identifier: int
        unique ID for customer

    env: simpy.Environment()
        current environment simulation is running in
        use this to pause & restart process after a delay

    args: Experiment
    '''

    start_wait = env.now

    #request an operator. parameters defined in Experiment
    with args.cell.request() as cell_req:
        yield cell_req

        outside_wait_time = env.now - start_wait
        args.results['outside_waiting_times'].append(outside_wait_time)

        if customer_type=="short":

            with args.operators.request() as req:
                yield req

                wait_time = env.now - start_wait

                args.results['waiting_times'].append(wait_time)

                call_duration = args.call_dist.sample()

                #sched process to begin after call_duration
                yield env.timeout(call_duration)

                args.results['total_short_transact_duration'] += call_duration

        elif customer_type =='long':
            with args.long_operators.request() as long_req:
                yield long_req
                
                wait_time = env.now - start_wait

                args.results['waiting_times'].append(wait_time)

                call_duration = args.long_call_dist.sample()

                #sched process to begin after call_duration
                yield env.timeout(call_duration)

                args.results['total_long_transact_duration'] += call_duration

        else:
            raise ValueError(f"Unknown customer type: {customer_type}")
        

def bell_curve(x, mu, sigma):

    return np.exp(-(x-mu)**2/(2*(sigma**2)))

def get_dynamic_mean_iat(current_time, short_iat, long_iat, peak_min, peak_min_standard_deviation):
    """
    Returns a dynamic mean inter-arrival time based on current simulation time.
    """

    mean_iat = short_iat + (long_iat-short_iat)*(1-bell_curve(current_time, peak_min, peak_min_standard_deviation))
    return mean_iat

def arriv_gen(env, args):

    for caller_count in itertools.count(start=1):
        # Get dynamic mean IAT based on current simulation time
        current_time = env.now
        dynamic_mean_iat = get_dynamic_mean_iat(current_time, args.short_iat, args.long_iat, args.peak_min, args.peak_min_standard_deviation)
        
        # Sample with updated mean
        inter_arrival_time = args.arrival_dist.sample(mean=dynamic_mean_iat)

        yield env.timeout(inter_arrival_time)

        env.process(service(caller_count, env, args, customer_type='short'))

def long_arriv_gen(env, args):
    
    for caller_count in itertools.count(start=1):
        # Get dynamic mean IAT based on current simulation time
        current_time = env.now
        dynamic_mean_iat = get_dynamic_mean_iat(current_time, args.long_transact_short_iat, args.long_transact_long_iat, args.peak_min, args.peak_min_standard_deviation)
        
        # Sample with updated mean
        inter_arrival_time = args.arrival_dist.sample(mean=dynamic_mean_iat)

        yield env.timeout(inter_arrival_time)

        env.process(service(caller_count, env, args, customer_type='long'))


def single_run(experiment, rep=0, rc_period=RESULTS_COLLECTION_PERIOD):
    #rep: replication number

    run_results = {}

    experiment.init_results_variables()

    experiment.set_random_no_set(rep)
    #each time you run replication number 2,
    #you run random number set 2

    env = simpy.Environment()
    experiment.operators = simpy.Resource(env, capacity=experiment.n_operators)
    experiment.cell = simpy.Resource(env, capacity=experiment.customer_capacity)
    experiment.long_operators = simpy.Resource(env, capacity=experiment.n_long_operators)

    env.process(arriv_gen(env, experiment))
    env.process(long_arriv_gen(env, experiment))
    env.run(until=rc_period)

    run_results['01_mean_wait_time'] = np.mean(experiment.results['waiting_times'])
    run_results['02_teller_util'] = (experiment.results['total_short_transact_duration'] / (rc_period * experiment.n_operators)) *100.0
    run_results['03_mean_outside_wait_time'] = np.mean(experiment.results['outside_waiting_times'])
    run_results['04_long_teller_util'] = (experiment.results['total_long_transact_duration'] / (rc_period * experiment.n_long_operators)) *100.0

    return run_results


def parallel_run_one_day(data, rep=0):

    #Runs multiple branches in parallel
    #Accepts dataframe of branches with parameters
    '''input:data must be cleaned for code to work(all columns in data as a df are variable names)
    
    output: dataframe of wait time, teller util, etc. the indexing of the branches from the input is followed'''

    branch_experiments = []

    for branch_id in range(len(data)):
        exp = Experiment(**data.iloc[branch_id].to_dict())  # customize per branch

        #run each single experiment
        xp = single_run(exp, rep=rep, rc_period=RESULTS_COLLECTION_PERIOD)

        branch_experiments.append(xp)


    ##Clean up output into dataframe
    branch_mean_wait_time_1d = []
    branch_teller_util_1d = []
    branch_mean_outside_wait_time_1d = []
    branch_long_teller_util_1d = []

    for k in branch_experiments:
        branch_mean_wait_time_1d.append(np.float64(k['01_mean_wait_time']))
        branch_teller_util_1d.append(np.float64(k['02_teller_util']))
        branch_mean_outside_wait_time_1d.append(np.float64(k['03_mean_outside_wait_time']))
        branch_long_teller_util_1d.append(np.float64(k['04_long_teller_util']))


    results_df = pd.DataFrame.from_dict({'01_mean_wait_time_1d':branch_mean_wait_time_1d ,
                                        '02_teller_util_1d':branch_teller_util_1d,
                                        '03_mean_outside_wait_time':branch_mean_outside_wait_time_1d,
                                        '04_long_teller_util_1d': branch_long_teller_util_1d
                                        })

    return results_df

def parallel_run_n_days(data, n_days, random_toggle=False):
    #random_toggle= False by default
    #if random_toggle is on, the run uses different seeds for each day

    ##seed sequence is fixed [0, 1, 2, ...] if random_toggle=False
    seed_sequence = [k for k in range(n_days)]
    if random_toggle:
        seed_sequence = np.random.randint(1, 100, size=n_days)

    n_rows = len(data)
    n_cols = N_METRICS

    save_data = np.zeros((n_days, n_rows, n_cols))

    for i in range(len(seed_sequence)):
        seed = seed_sequence[i]
        day_df = parallel_run_one_day(data, rep=seed) #use seed from seed sequence to introduce variation per day
        save_data[i, : , 0] = day_df['01_mean_wait_time_1d'] 
        save_data[i, : , 1] = day_df['02_teller_util_1d']
        save_data[i, : , 2] = day_df['03_mean_outside_wait_time']
        save_data[i, : , 3] = day_df['04_long_teller_util_1d']

    return save_data

def single_run_n_days(exp, n_days, random_toggle=False):
    '''Runs a single branch for n_days.
    experiment= Experiment(*parameters specified).
    random_toggle=False by default. A fixed seed is used if random_toggle=False. 
    If random_toggle=True, a seed sequence is generated and used.
    Each seed in the seed_sequence is used in the replication value (rep) of the single_run command.'''
    #random_toggle= False by default
    #if random_toggle is on, the run uses different seeds for each day

    ##seed sequence is fixed [0, 1, 2, ...] if random_toggle=False
    seed_sequence = [k for k in range(n_days)]
    if random_toggle:
        seed_sequence = np.random.randint(1, 100, size=n_days)

    run_dict = []
    
    
    for i in range(len(seed_sequence)):
        seed = seed_sequence[i]
        
        run_dict.append(single_run(exp,rep=seed))

    run_dict_transform = {'01_mean_wait_time': [k['01_mean_wait_time'] for k in run_dict],
                          '02_teller_util': [k['02_teller_util'] for k in run_dict],
                          '03_mean_outside_wait_time': [k['03_mean_outside_wait_time'] for k in run_dict],
                          '04_long_teller_util': [k['04_long_teller_util'] for k in run_dict]}

    return pd.DataFrame.from_dict(run_dict_transform)
    

