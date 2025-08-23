import numpy as np
import simpy
import pandas as pd
import itertools

# default resources
N_OPERATORS = 2
N_LONG_OPERATORS = 2 #Long transactions such as loan applications and financial consultation


#Queueing behavior parameters
#for get_dynamic_mean_iat(current_time, PEAK_MIN, PEAK_MIN_STANDARD_DEVIATION, LONG_IAT, SHORT_IAT):
PEAK_MIN= 4*60 #minute at which peak time is, measured from opening hour (e.g. 180 mins from opening hours)
PEAK_MIN_STANDARD_DEVIATION= 3*60 #smaller PEAK_MIN_STANDARD_DEVIATION = much much higher traffic during peak hr

##Inter-arrival time
BASE_MEAN_IAT = 4.0 # default mean inter-arrival time (exp)
BASE_VAR_IAT =  0.5 #difference between mean inter-arrival time and highest inter-arrival time or lowest inter-arrival time
LONG_IAT =      BASE_MEAN_IAT+2*BASE_VAR_IAT
SHORT_IAT=      BASE_MEAN_IAT-BASE_VAR_IAT
#percent of long transactions that consist of all transactions processed by the branch

LONG_TRANSACT_BASE_MEAN_IAT = 75.0    #CONTROLLABLE
LONG_TRANSACT_BASE_VAR_IAT = 35.0     #CONTROLLABLE
LONG_TRANSACT_LONG_IAT =    LONG_TRANSACT_BASE_MEAN_IAT + 2*LONG_TRANSACT_BASE_VAR_IAT        
LONG_TRANSACT_SHORT_IAT =   LONG_TRANSACT_BASE_MEAN_IAT - LONG_TRANSACT_BASE_VAR_IAT

#simulate queueing behavior inside branch
#Suppose branches have a particular capacity for a particular queue

AREA_BRANCH =               150.0 #in sq. m 
AREA_WAIT_RATIO =           0.05 #this should be a 0-> 1 slider in streamlit
AREA_WAIT =                 AREA_WAIT_RATIO*AREA_BRANCH
CUSTOMER_PREFERRED_AREA =   1 #suppose a customer likes having 1 sq. m of space while waiting
#in units of sq. m per person
#Make this adjustable also. The default should be estimated

CUSTOMER_CAPACITY = int(np.round(AREA_WAIT/CUSTOMER_PREFERRED_AREA)) #calculated customer capacity based on available waiting area and 
#preferred personal space of customer




# default service time parameters (triangular)
#This is for short transactions
CALL_LOW = 5.0
CALL_MODE = 7.0
CALL_HIGH = 10.0

#This is for long transactions
LONG_CALL_LOW = 60
LONG_CALL_MODE = 70
LONG_CALL_HIGH = 100

# sampling settings
N_STREAMS = 4 #DO NOT change
DEFAULT_RND_SET = 0
LOW_UNIF = 1
HIGH_UNIF = 100

# Boolean switch to simulation results as the model runs
TRACE = False

# run variables
RESULTS_COLLECTION_PERIOD = 60*8
N_METRICS = 4 #we have mean_wait_time, teller_util, outside_mean_wait_time, long_teller_util

class Uniform:
    def __init__(self, low, high, random_seed = None):
        self.rand = np.random.default_rng(seed = random_seed)
        self.low = low
        self.high = high
    
    def sample(self, size=None):
        return self.rand.integers(self.low, self.high, size=size)

class Triangular:
    def __init__(self, low, mode, high, random_seed = None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.mode = mode
        self.high = high
    
    def sample(self, size=None):
        """
        Draws a sample from an exponential distribution.
        
        Parameters:
        - size: int or tuple, optional — output shape
        - mean: float, optional — overrides default_mean for this sample

        Returns:
        - float or ndarray
        """
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
        """
        Draws a sample from an exponential distribution.
        
        Parameters:
        - size: int or tuple, optional — output shape
        - mean: float, optional — overrides default_mean for this sample

        Returns:
        - float or ndarray
        """
        return self.rand.normal(loc=self.mean, scale=self.std_dev, size=size)
    
class daily_rand:
    def __init__(
        self,
        random_number_set=  DEFAULT_RND_SET,
        low_unif = LOW_UNIF,
        high_unif = HIGH_UNIF,
        n_streams=  N_STREAMS 
    ):
        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams
        self.low_unif = low_unif
        self.high_unif = high_unif
        
        # store parameters for the run of the model
        
    
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
        self.daily_rand = Uniform(self.low_unif, self.high_unif, random_seed=self.seeds[0])

    
class Experiment:
    """
    Encapsulates the concept of an experiment.

    An Experiment:
    1. Contains a list of parameters that can be left as defaults or varied
    2. Provides a place for the experimentor to record results of a run 
    3. Controls the set & streams of psuedo random numbers used in a run.
    
    """

    def __init__(
        self,
        random_number_set=  DEFAULT_RND_SET,
        n_operators=        N_OPERATORS,
        n_long_operators=   N_LONG_OPERATORS,
        base_mean_iat =     BASE_MEAN_IAT,
        base_var_iat  =     BASE_VAR_IAT,
        short_iat =         None, #BASE_MEAN_IAT - BASE_VAR_IAT
        long_iat =          None, #BASE_MEAN_IAT + 2*BASE_VAR_IAT,
        
        long_call_low =     LONG_CALL_LOW,
        long_call_mode =    LONG_CALL_MODE,
        long_call_high=     LONG_CALL_HIGH,
        call_low=           CALL_LOW,
        call_mode=          CALL_MODE,
        call_high=          CALL_HIGH,
        
        peak_min =          PEAK_MIN,
        long_transact_base_mean_iat =   LONG_TRANSACT_BASE_MEAN_IAT,
        long_transact_base_var_iat =    LONG_TRANSACT_BASE_VAR_IAT,
        long_transact_short_iat =       None, #LONG_TRANSACT_BASE_MEAN_IAT - LONG_TRANSACT_BASE_VAR_IAT, 
        long_transact_long_iat =        None, #LONG_TRANSACT_BASE_MEAN_IAT + 2*LONG_TRANSACT_BASE_VAR_IAT,
        peak_min_standard_deviation =   PEAK_MIN_STANDARD_DEVIATION,
        n_streams=                      N_STREAMS,
        customer_capacity =             CUSTOMER_CAPACITY, #include mechanism to just calculate customer capacity in the website
    ):
        #get_dynamic_mean_iat(current_time, PEAK_MIN, PEAK_MIN_STANDARD_DEVIATION, LONG_IAT, SHORT_IAT):

        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams
        
        # store parameters for the run of the model
        self.n_operators = n_operators
        self.n_long_operators = n_long_operators

        #
        self.base_mean_iat = base_mean_iat
        self.base_var_iat = base_var_iat
        self.short_iat = short_iat if short_iat is not None else base_mean_iat - base_var_iat
        self.long_iat = long_iat if short_iat is not None else base_mean_iat + base_var_iat

        self.long_transact_base_mean_iat = long_transact_base_mean_iat
        self.long_transact_base_var_iat = long_transact_base_var_iat
        self.long_transact_long_iat = long_transact_long_iat if long_transact_long_iat is not None else long_transact_base_mean_iat + long_transact_base_var_iat
        self.long_transact_short_iat = long_transact_short_iat if long_transact_short_iat is not None else long_transact_base_mean_iat - long_transact_base_var_iat

        self.call_low =         call_low
        self.call_mode =        call_mode
        self.call_high =        call_high
        self.long_call_low =    long_call_low
        self.long_call_mode = long_call_mode
        self.long_call_high = long_call_high

        self.customer_capacity = customer_capacity
    
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

        if self.base_var_iat >= self.base_mean_iat:
            raise ValueError(
                f"`base_var_iat` ({self.base_var_iat}) must be smaller than `base_mean_iat` ({self.base_mean_iat})")
        
        if self.long_transact_base_var_iat >= self.long_transact_base_mean_iat:
            raise ValueError(
                f"`long_transact_base_var_iat` ({self.long_transact_base_var_iat }) must be smaller than `long_transact_base_mean_iat` ({self.long_transact_base_mean_iat })")

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
            self.base_mean_iat, random_seed=self.seeds[0]
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
        self.results['short_transact_time_of_arrival'] = []
        self.results["short_transact_waiting_times"] = []

        self.results['long_transact_time_of_arrival'] = []
        self.results['long_transact_waiting_times'] = []

        self.results['outside_waiting_times_time_of_arrival'] = []
        self.results['outside_waiting_times'] = []

        # total operator usage time for utilisation calculation.
        self.results["total_short_transact_duration"] = 0.0
        self.results['total_long_transact_duration'] = 0.0

def periodic_exp(x_arr, b, k, max_day, rate):
    y = (b-k)*np.exp((x_arr-max_day)/rate)+k
    return y

def rescale_exp(force_max, force_min, max_day, rate):
    """Rescales the parameters for the exponential function based on the desired maximum value and minimum value at specific x.
    Solved this by hand from the periodic exponential function."""
    b = force_max
    k = (force_min-b*np.exp(-max_day/rate))/(1-np.exp(-max_day/rate))
    return b, k, max_day, rate

def salary_iat_variation(last_day_of_month=30, force_max=5, force_min=3, rate=5):
    """Generates a list of varying daily mean inter-arrival times.
    Based on the behavior that more short transactions occur on and after the 15th and the 30th."""

    calendar_days = np.array([k+1 for k in range(last_day_of_month)])
    blank_arr = np.zeros(last_day_of_month)
    peak_days = [0, 15, last_day_of_month]
    select1= np.where((calendar_days > peak_days[0])*(calendar_days < peak_days[1]))
    select2 = np.where((calendar_days > peak_days[1]-1)*(calendar_days < peak_days[2]))
    select3 = np.where((calendar_days >= peak_days[2]))

    blank_arr[select1] = periodic_exp(calendar_days[select1], *rescale_exp(force_max, force_min, peak_days[1]-1, rate=rate))
    blank_arr[select2] = periodic_exp(calendar_days[select2], *rescale_exp(force_max, force_min, peak_days[2]-1, rate=rate))
    blank_arr[select3] = periodic_exp(calendar_days[select3], *rescale_exp(force_max, force_min, peak_days[2]-1 + 15, rate=rate))
    #blank_arr[peak_days[0]] = force_min
    blank_arr[peak_days[1]-1] = force_min
    if peak_days[2] == 30:
        blank_arr[peak_days[2]-1] = force_min 
    elif peak_days[2] == 31:
        blank_arr[peak_days[2]-2] = force_min
    elif peak_days[2] == 28:
        blank_arr[peak_days[2]-1] = force_min
    elif peak_days[2] == 29:
        blank_arr[-1] = force_min
    return calendar_days, blank_arr

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

    if customer_type=="short":

        with args.cell.request() as cell_req:
            yield cell_req

            outside_wait_time = env.now - start_wait
            args.results['outside_waiting_times'].append(outside_wait_time)
            args.results['outside_waiting_times_time_of_arrival'].append(start_wait)

            with args.operators.request() as req:
                yield req

                wait_time = env.now - start_wait

                args.results['short_transact_waiting_times'].append(wait_time) 
                #transaction waiting times are inclusive of the outside waiting time
                args.results['short_transact_time_of_arrival'].append(start_wait)

                call_duration = args.call_dist.sample()

                #sched process to begin after call_duration
                yield env.timeout(call_duration)

                args.results['total_short_transact_duration'] += call_duration
            
    elif customer_type =='long':
        
        with args.cell.request() as cell_req:
            yield cell_req

            outside_wait_time = env.now - start_wait
            args.results['outside_waiting_times'].append(outside_wait_time)
            args.results['outside_waiting_times_time_of_arrival'].append(start_wait)    

            with args.long_operators.request() as long_req:
                yield long_req
                
                wait_time = env.now - start_wait

                args.results['long_transact_waiting_times'].append(wait_time)
                #long transaction waiting times are inclusive of outside waiting times
                args.results['long_transact_time_of_arrival'].append(start_wait)


                call_duration = args.long_call_dist.sample()

                #sched process to begin after call_duration
                yield env.timeout(call_duration)

                args.results['total_long_transact_duration'] += call_duration

    else:
        raise ValueError(f"Unknown customer type: {customer_type}")
        
def bell_curve(x, mu, sigma):

    return np.exp(-(x-mu)**2/(2*(sigma**2)))

def get_dynamic_mean_iat(current_time_, base_mean_iat_, base_var_iat_, peak_min_, peak_min_stddev_):
    """
    Daily oscillation around a base mean inter-arrival time.
    base_var_iat_ is the amplitude of oscillation.
    """
    fluctuation = base_var_iat_ * (1 - bell_curve(current_time_, peak_min_, peak_min_stddev_))
    return max(0.01, base_mean_iat_ + fluctuation)

def arriv_gen(env, args):
    
    for caller_count in itertools.count(start=1):
        # Get dynamic mean IAT based on current simulation time
        current_time = env.now
        dynamic_mean_iat = get_dynamic_mean_iat(
            current_time,
            args.base_mean_iat,
            args.base_var_iat,
            args.peak_min,
            args.peak_min_standard_deviation)

        # Sample with updated mean
        inter_arrival_time = args.arrival_dist.sample(mean=dynamic_mean_iat)

        yield env.timeout(inter_arrival_time)

        env.process(service(caller_count, env, args, customer_type='short'))

def long_arriv_gen(env, args):
    
    for caller_count in itertools.count(start=1):
        # Get dynamic mean IAT based on current simulation time
        current_time = env.now
        dynamic_mean_iat = get_dynamic_mean_iat(current_time, 
                                                args.long_transact_short_iat, 
                                                args.long_transact_long_iat, 
                                                args.peak_min, 
                                                args.peak_min_standard_deviation)
        
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

    #results that summarize the run for the day
    run_results['01_mean_wait_time'] = np.mean(experiment.results['short_transact_waiting_times'])
    run_results['02_teller_util'] = (experiment.results['total_short_transact_duration'] / (rc_period * experiment.n_operators)) *100.0
    run_results['03_mean_outside_wait_time'] = np.mean(experiment.results['outside_waiting_times'])
    run_results['04_long_teller_util'] = (experiment.results['total_long_transact_duration'] / (rc_period * experiment.n_long_operators)) *100.0
    
    #parameters for plotting time series data
    run_results['05_short_transact_wait_times'] = experiment.results['short_transact_waiting_times']
    run_results['06_short_transact_time_of_arrival'] = experiment.results['short_transact_time_of_arrival']
    run_results['07_outside_wait_times'] = experiment.results['outside_waiting_times']
    run_results['08_outside_wait_times_time_of_arrival'] = experiment.results['outside_waiting_times_time_of_arrival']
    run_results['09_long_transact_wait_times'] = experiment.results['long_transact_waiting_times']
    run_results['10_long_transact_wait_times_time_of_arrival'] = experiment.results['long_transact_time_of_arrival']

    return run_results

def month_run(blank_xp, 
              last_day_of_month, 
              force_max, 
              force_min, 
              rate, 
              seed = 0, 
              rc_period = RESULTS_COLLECTION_PERIOD):
    
    run_days = [k for k in range(last_day_of_month)]
    rand = daily_rand()
    rand.set_random_no_set(seed)

    run_seed = [rand.daily_rand.sample() for k in run_days]

    base_iats = salary_iat_variation(last_day_of_month=last_day_of_month, 
                                     force_max=force_max, 
                                     force_min=force_min, 
                                     rate=rate)[1]

    run_results = {}
    run_results['01_1month_mean_wait_time'] = []
    run_results['02_1month_teller_util'] = []
    run_results['03_1month_mean_outside_wait_time'] = []
    run_results['04_1month_long_teller_util'] = []

    exclude_vars = ['base_mean_iat', 'operators', 'long_operators', 'cell', 'results', 'seeds', 'arrival_dist', 'call_dist',
                    'long_call_dist']

    ddd = {key: value for key, value in blank_xp.__dict__.items() if key not in exclude_vars}

    for k in range(len(run_days)):
        experiment = Experiment(**ddd, base_mean_iat=base_iats[k])
        experiment.init_results_variables()
        experiment.set_random_no_set(run_seed[k])

        env = simpy.Environment()

        experiment.init_results_variables()

        experiment.set_random_no_set(run_seed[k])
        #each time you run replication number 2,
        #you run random number set 2
        experiment.base_mean_iat = base_iats[k]

        env = simpy.Environment()
        experiment.operators = simpy.Resource(env, capacity=experiment.n_operators)
        experiment.cell = simpy.Resource(env, capacity=experiment.customer_capacity)
        experiment.long_operators = simpy.Resource(env, capacity=experiment.n_long_operators)

        env.process(arriv_gen(env, experiment))
        env.process(long_arriv_gen(env, experiment))
        env.run(until=rc_period)

        #results that summarize the run for the day
        run_results['01_1month_mean_wait_time'].append(np.mean(experiment.results['short_transact_waiting_times']))
        run_results['02_1month_teller_util'].append(experiment.results['total_short_transact_duration'] / (rc_period * experiment.n_operators)*100.0)
        run_results['03_1month_mean_outside_wait_time'].append(np.mean(experiment.results['outside_waiting_times']))
        run_results['04_1month_long_teller_util'].append(experiment.results['total_long_transact_duration'] / (rc_period * experiment.n_long_operators)*100.0)


    return run_results



    