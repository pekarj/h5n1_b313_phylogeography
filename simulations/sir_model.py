"""
Unified county-level SIR model for H5N1 B3.13 cattle epidemic simulation.

Provides network loading, recovery computation, and the SIR simulation
engine with a flexible NPI layer system. All other scripts import from here.
"""

import math
import hashlib
import random
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml


##### Config loading #####

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(config, key):
    """Resolve a data path relative to the config file's directory."""
    base = os.path.dirname(os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'config.yaml')))
    return os.path.normpath(os.path.join(base, config['data'][key]))


##### State FIPS mapping #####

STATE_FIPS_TO_ABBR = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '12': 'FL', '13': 'GA',
    '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
    '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO',
    '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ',
    '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC',
    '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT',
    '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY',
}

ABBR_TO_STATE_FIPS = {v: k for k, v in STATE_FIPS_TO_ABBR.items()}


def county_to_state(county_fips):
    prefix = str(county_fips).zfill(5)[:2]
    return STATE_FIPS_TO_ABBR.get(prefix)


def build_county_state_map(Sz):
    return {county: county_to_state(county) for county in Sz}


##### Data loading #####

def load_county_data(config):
    path = resolve_path(config, 'county_data')
    dc = pd.read_csv(path)
    dc['County'] = dc['County'].astype(int)
    Sz = dict(zip(dc['County'], dc['d']))
    Pz = dict(zip(dc['County'], dc['p']))
    return Sz, Pz


def compute_recovery_periods(Sz, Pz, alpha, gamma, mu_c):
    mu_C = {}
    for cid in Sz:
        mu_C[cid] = round(mu_c * (1 + alpha * math.log(Sz[cid])
                                  * (1 + gamma * math.log(Pz[cid]))))
    return mu_C


def load_network(filepath):
    """Load a USAMMv3 dairy network. Returns {day_of_year: {(o, d): volume}}."""
    dairy_net = pd.read_csv(
        filepath, sep='\t',
        usecols=['dayOfYear', 'oCountyId', 'dCountyId', 'volume'],
        dtype={'dayOfYear': np.int32, 'oCountyId': np.int32,
               'dCountyId': np.int32, 'volume': np.float64}
    )
    dairy_net = dairy_net[
        (dairy_net['oCountyId'] != dairy_net['dCountyId']) &
        (dairy_net['dayOfYear'] >= 1) & (dairy_net['dayOfYear'] <= 365)
    ]
    grouped = dairy_net.groupby(
        ['dayOfYear', 'oCountyId', 'dCountyId'], sort=False
    )['volume'].sum()

    daily_edges = {d: {} for d in range(1, 366)}
    for (d, o, dest), vol in grouped.items():
        daily_edges[d][(o, dest)] = vol
    return daily_edges


def build_adj_index(daily_edges):
    """Pre-index edges by day and origin for fast lookup."""
    adj_index = {}
    for d, edges in daily_edges.items():
        adj = {}
        for (o, dest), vol in edges.items():
            adj.setdefault(o, []).append((dest, vol))
        adj_index[d] = adj
    return adj_index


def get_sorted_network_paths(config):
    network_dir = resolve_path(config, 'network_dir')
    return sorted([
        os.path.join(network_dir, f)
        for f in os.listdir(network_dir)
        if f.startswith('dairy_network_') and f.endswith('.network')
    ])


##### Timeline helpers #####

def get_tmrca_doy(tmrca_offset, tx_detection_doy=85):
    doy = (tx_detection_doy - tmrca_offset) % 365
    return doy if doy > 0 else 365


##### NPI layer system #####

class NpiLayer:
    """A single NPI layer that can reduce edge weights when active."""

    def __init__(self, target, factor, activation_day, county_state,
                 states=None, county_set=None):
        """
        Args:
            target: 'interstate', 'intrastate', 'from_states', 'from_counties',
                    'to_states', 'all'
            factor: float in [0, 1] — multiplicative reduction
            activation_day: sim day when this layer activates
            county_state: dict {county_fips: state_abbr}
            states: list of state abbreviations (for from_states/to_states)
            county_set: set of county FIPS codes (for from_counties)
        """
        self.target = target
        self.factor = factor
        self.activation_day = activation_day
        self.county_state = county_state
        self._states = set(states) if states else set()
        self._county_set = set(county_set) if county_set else set()

    def is_active(self, sim_day):
        return sim_day >= self.activation_day

    def matches_edge(self, src, dest):
        if self.target == 'interstate':
            return self.county_state.get(src) != self.county_state.get(dest)
        elif self.target == 'intrastate':
            return self.county_state.get(src) == self.county_state.get(dest)
        elif self.target == 'from_states':
            return self.county_state.get(src) in self._states
        elif self.target == 'to_states':
            return self.county_state.get(dest) in self._states
        elif self.target == 'from_counties':
            return src in self._county_set
        elif self.target == 'all':
            return True
        return False


def build_npi_layers(npi_defs, tmrca_offset, county_state, posterior_npi_factor,
                     config, county_sets=None):
    """Build NpiLayer objects from scenario NPI definitions.

    Args:
        npi_defs: list of dicts from config (target, factor, delay_post_tx/delay_post_tmrca, ...)
        tmrca_offset: days from tMRCA to TX detection for this particle
        county_state: dict {county_fips: state_abbr}
        posterior_npi_factor: float — the ABC posterior npi_factor value
        config: full config dict
        county_sets: dict of named county sets (optional)
    """
    tx_to_npi = config['timeline']['tx_to_npi_days']
    layers = []

    for npi_def in npi_defs:
        target = npi_def['target']

        # Resolve factor
        factor = npi_def['factor']
        if factor == 'posterior':
            factor = posterior_npi_factor

        # Resolve activation day
        if 'delay_post_tx' in npi_def:
            activation_day = tmrca_offset + npi_def['delay_post_tx']
        elif 'delay_post_tmrca' in npi_def:
            activation_day = npi_def['delay_post_tmrca']
        else:
            activation_day = 0

        # Resolve target-specific params
        states = npi_def.get('states')
        county_set = None
        if 'county_set' in npi_def:
            cs_name = npi_def['county_set']
            if county_sets and cs_name in county_sets:
                county_set = county_sets[cs_name]
            elif 'county_sets' in config and cs_name in config['county_sets']:
                county_set = config['county_sets'][cs_name]

        layers.append(NpiLayer(
            target=target,
            factor=factor,
            activation_day=activation_day,
            county_state=county_state,
            states=states,
            county_set=county_set,
        ))

    return layers


def compute_npi_factor(layers, src, dest, sim_day):
    """Compute multiplicative NPI factor for an edge on a given day."""
    combined = 1.0
    for layer in layers:
        if layer.is_active(sim_day) and layer.matches_edge(src, dest):
            combined *= layer.factor
    return combined


##### Deterministic per-simulation seeding #####

def make_sim_seed(master_seed, param_idx, net_idx, sim_idx, scenario_name):
    key = f"{master_seed}:{param_idx}:{net_idx}:{sim_idx}:{scenario_name}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


##### SIR simulation #####

def simulate_sir(adj_index, Sz, mu_C, beta, seeds, num_days,
                 npi_layers, tmrca_doy, rng=None):
    """Run one stochastic SIR simulation with NPI layers.

    Args:
        adj_index: pre-indexed adjacency {day_of_year: {origin: [(dest, vol), ...]}}
        Sz: dict of county dairy cow counts (keys = valid counties)
        mu_C: dict {county: recovery_period_days}
        beta: transmission coefficient
        seeds: list of seed county FIPS codes
        num_days: simulation duration
        npi_layers: list of NpiLayer objects (can be empty)
        tmrca_doy: day-of-year for sim day 0
        rng: random.Random instance for reproducibility

    Returns:
        (infected_time, transmission_log)
        infected_time: dict {county_fips: sim_day_of_infection}
        transmission_log: list of (src_county, dest_county, sim_day)
    """
    if rng is None:
        rng = random.Random()

    valid_counties = set(Sz.keys())
    S = set(valid_counties)
    I = set()
    infected_time = {}
    transmission_log = []

    for s in seeds:
        if s in S:
            I.add(s)
            S.remove(s)
            infected_time[s] = 0

    has_npis = len(npi_layers) > 0

    for sim_day in range(num_days):
        network_doy = (tmrca_doy + sim_day - 1) % 365 + 1
        adj_today = adj_index.get(network_doy, {})

        new_infections = set()
        new_recoveries = set()

        for u in list(I):
            if sim_day >= infected_time[u] + mu_C.get(u, 14):
                new_recoveries.add(u)
                continue

            for dest, vol in adj_today.get(u, []):
                if dest in S and dest in valid_counties:
                    w = vol
                    if has_npis:
                        npi_f = compute_npi_factor(npi_layers, u, dest, sim_day)
                        w *= npi_f
                    inf_prob = 1 - math.exp(-beta * w)
                    if rng.random() < inf_prob:
                        new_infections.add(dest)
                        S.remove(dest)
                        transmission_log.append((u, dest, sim_day))

        for v in new_infections:
            I.add(v)
            infected_time[v] = sim_day
        for w in new_recoveries:
            I.discard(w)

    return infected_time, transmission_log


##### Summary statistics #####

def extract_state_arrivals(infection_times):
    """First arrival day per state (sim days since tMRCA)."""
    state_days = {}
    for county, day in infection_times.items():
        state = county_to_state(county)
        if state is None:
            continue
        if state not in state_days or day < state_days[state]:
            state_days[state] = day
    return state_days


def extract_state_county_counts(infection_times):
    """Number of infected counties per state."""
    state_counts = {}
    for county in infection_times:
        state = county_to_state(county)
        if state is None:
            continue
        state_counts[state] = state_counts.get(state, 0) + 1
    return state_counts


def classify_transmissions(transmission_log, npi_sim_day, county_state):
    """Classify transmissions as pre/post NPI and interstate/intrastate."""
    pre_interstate = 0
    pre_intrastate = 0
    post_interstate = 0
    post_intrastate = 0

    for src, dest, sim_day in transmission_log:
        is_interstate = (county_state.get(src) != county_state.get(dest))
        is_pre_npi = (sim_day < npi_sim_day) if npi_sim_day else True

        if is_pre_npi:
            if is_interstate:
                pre_interstate += 1
            else:
                pre_intrastate += 1
        else:
            if is_interstate:
                post_interstate += 1
            else:
                post_intrastate += 1

    return {
        'pre_interstate': pre_interstate,
        'pre_intrastate': pre_intrastate,
        'post_interstate': post_interstate,
        'post_intrastate': post_intrastate,
    }


def compute_state_metrics(infection_times, transmission_log, county_state):
    """Per-state metrics: arrivals, introductions, intrastate infections, county counts."""
    state_introductions = defaultdict(int)
    state_intrastate = defaultdict(int)
    state_infected_counties = defaultdict(set)

    for src, dest, sim_day in transmission_log:
        src_st = county_state.get(src)
        dest_st = county_state.get(dest)
        if src_st != dest_st:
            state_introductions[dest_st] += 1
        else:
            state_intrastate[dest_st] += 1

    for county in infection_times:
        st = county_state.get(county)
        if st:
            state_infected_counties[st].add(county)

    state_arrivals = extract_state_arrivals(infection_times)

    return {
        'state_arrivals': state_arrivals,
        'state_introductions': dict(state_introductions),
        'state_intrastate': dict(state_intrastate),
        'state_infected_counties': {st: len(cs)
                                    for st, cs in state_infected_counties.items()},
    }


##### Connectivity pools (for alternative origin analysis) #####

def build_connectivity_pools(Sz, config, n_sample_nets=20):
    """Compute per-state connectivity pools (high/low quartile by out-degree).

    Returns:
        pools: {state_abbr: {'high': [fips, ...], 'low': [fips, ...]}}
        county_degree: {fips: mean_out_degree}
    """
    network_dir = resolve_path(config, 'network_dir')
    all_nets = sorted([
        os.path.join(network_dir, f)
        for f in os.listdir(network_dir)
        if f.startswith('dairy_network_') and f.endswith('.network')
    ])
    step = max(1, len(all_nets) // n_sample_nets)
    sample_nets = all_nets[::step][:n_sample_nets]

    print(f"  Building connectivity pools from {len(sample_nets)} sampled networks...",
          flush=True)

    degree_accum = defaultdict(list)
    for f in sample_nets:
        df = pd.read_csv(f, sep='\t')
        out_deg = df.groupby('oCountyId')['dCountyId'].nunique()
        for fips, deg in out_deg.items():
            degree_accum[fips].append(deg)

    county_degree = {}
    for fips in Sz:
        county_degree[fips] = np.mean(degree_accum.get(fips, [0]))

    seed_states = config['alternative_origins']['seed_states']
    state_counties = defaultdict(list)
    for fips in Sz:
        st = county_to_state(fips)
        if st:
            state_counties[st].append(fips)

    pools = {}
    for st in seed_states:
        counties = state_counties.get(st, [])
        if len(counties) == 0:
            print(f"  WARNING: no dairy counties for {st}", flush=True)
            pools[st] = {'high': [], 'low': []}
            continue

        degrees = np.array([county_degree[c] for c in counties])
        if len(counties) < 4:
            pools[st] = {'high': counties[:], 'low': counties[:]}
        else:
            q25 = np.percentile(degrees, 25)
            q75 = np.percentile(degrees, 75)
            high = [c for c, d in zip(counties, degrees) if d >= q75]
            low = [c for c, d in zip(counties, degrees) if d <= q25]
            pools[st] = {'high': high, 'low': low}

        print(f"    {st}: {len(counties)} dairy counties, "
              f"high={len(pools[st]['high'])}, low={len(pools[st]['low'])}",
              flush=True)

    return pools, county_degree
