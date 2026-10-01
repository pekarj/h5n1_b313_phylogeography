"""
Downstream simulation runner — all scenarios from config.yaml.

Replaces run_counterfactual_timing.py, run_alternative_origin.py, and
run_within_state_npi.py with a single config-driven script.

After each batch completes, transmission logs and county_times are
aggregated into compact summaries and the raw files are deleted to
conserve disk space.

Usage:
    python run_simulations.py --abc_results outputs/abc_final/abc_accepted.csv
    python run_simulations.py --groups npi_timing
    python run_simulations.py --groups targeted_containment --scenarios castro_tmrca14
    python run_simulations.py --groups alternative_origins
    python run_simulations.py --num_networks 50  # quick validation
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sir_model import (
    load_config, resolve_path, load_county_data, load_network,
    build_adj_index, compute_recovery_periods, simulate_sir,
    build_npi_layers, build_connectivity_pools,
    extract_state_arrivals, classify_transmissions, compute_state_metrics,
    build_county_state_map, get_tmrca_doy, make_sim_seed, county_to_state,
)

from aggregate import (
    aggregate_state_pairs, aggregate_weekly_profiles,
    compute_chain_depth, classify_consequential,
)


##### Batch I/O #####

class ScenarioBatchWriter:
    """Streams transmission log and county times to CSV per scenario-batch."""

    TRANS_HEADER = ['param_idx', 'net_idx', 'sim_idx', 'scenario',
                    'seed_county', 'src_county', 'dest_county', 'sim_day']
    COUNTY_HEADER = ['param_idx', 'net_idx', 'sim_idx', 'scenario',
                     'seed_county', 'county', 'infection_day', 'recovery_day']

    def __init__(self, output_dir, batch_idx, scenario_name):
        batch_dir = os.path.join(output_dir, 'batches')
        os.makedirs(batch_dir, exist_ok=True)
        tag = f'batch_{batch_idx:03d}_{scenario_name}'

        self._trans_path = os.path.join(batch_dir, f'{tag}_transmission_log.csv')
        self._county_path = os.path.join(batch_dir, f'{tag}_county_times.csv')

        self._trans_file = open(self._trans_path, 'w', newline='')
        self._county_file = open(self._county_path, 'w', newline='')

        self._trans_writer = csv.writer(self._trans_file)
        self._county_writer = csv.writer(self._county_file)

        self._trans_writer.writerow(self.TRANS_HEADER)
        self._county_writer.writerow(self.COUNTY_HEADER)

        self._scenario_name = scenario_name

    def write_sim(self, param_idx, net_idx, sim_idx, seed_county,
                  transmission_log, infection_times, mu_C):
        for src, dest, day in transmission_log:
            self._trans_writer.writerow([
                param_idx, net_idx, sim_idx, self._scenario_name,
                seed_county, src, dest, day])
        for county, inf_day in infection_times.items():
            rec_day = inf_day + mu_C.get(county, 14)
            self._county_writer.writerow([
                param_idx, net_idx, sim_idx, self._scenario_name,
                seed_county, county, inf_day, rec_day])

    def flush(self):
        self._trans_file.flush()
        self._county_file.flush()

    def close(self):
        self._trans_file.close()
        self._county_file.close()


def save_batch_results(output_dir, batch_idx, scenario_name, results):
    batch_dir = os.path.join(output_dir, 'batches')
    os.makedirs(batch_dir, exist_ok=True)
    tag = f'batch_{batch_idx:03d}_{scenario_name}'
    pd.DataFrame(results).to_csv(
        os.path.join(batch_dir, f'{tag}.csv'), index=False)


def save_batch_state_detail(output_dir, batch_idx, scenario_name, state_rows):
    batch_dir = os.path.join(output_dir, 'batches')
    os.makedirs(batch_dir, exist_ok=True)
    tag = f'batch_{batch_idx:03d}_{scenario_name}_states'
    pd.DataFrame(state_rows).to_csv(
        os.path.join(batch_dir, f'{tag}.csv'), index=False)


def get_completed_scenario_batches(output_dir, scenario_names):
    """Check which (batch_idx, scenario) pairs are complete.

    A pair is complete if it has results + states files, AND either
    raw logs still exist OR aggregated files exist (logs were processed).
    """
    batch_dir = os.path.join(output_dir, 'batches')
    if not os.path.exists(batch_dir):
        return set()
    files = set(os.listdir(batch_dir))
    completed = set()
    for scenario_name in scenario_names:
        idx = 0
        while True:
            tag = f'batch_{idx:03d}_{scenario_name}'
            core_files = [f'{tag}.csv', f'{tag}_states.csv']
            raw_files = [f'{tag}_transmission_log.csv', f'{tag}_county_times.csv']
            agg_marker = f'{tag}_aggregated.done'

            if all(f in files for f in core_files):
                if all(f in files for f in raw_files) or agg_marker in files:
                    completed.add((idx, scenario_name))
                    idx += 1
                else:
                    break
            else:
                break
    return completed


##### Per-batch aggregation and cleanup #####

def aggregate_and_cleanup_batch(output_dir, batch_idx, scenario_name,
                                 county_state, tmrca_offsets, tx_to_npi_days,
                                 seed_county_default):
    """Aggregate transmission logs and county_times for one batch/scenario.

    Saves one small aggregate file per batch/scenario (fixed size, not appended).
    Raw transmission logs and county_times are deleted after aggregation.
    Final merge of per-batch aggregates happens in a separate step.
    """
    batch_dir = os.path.join(output_dir, 'batches')
    tag = f'batch_{batch_idx:03d}_{scenario_name}'
    trans_path = os.path.join(batch_dir, f'{tag}_transmission_log.csv')
    county_path = os.path.join(batch_dir, f'{tag}_county_times.csv')

    if not os.path.exists(trans_path):
        return

    trans_df = pd.read_csv(trans_path)
    county_df = pd.read_csv(county_path) if os.path.exists(county_path) else None

    if len(trans_df) == 0:
        os.remove(trans_path)
        if county_df is not None:
            os.remove(county_path)
        open(os.path.join(batch_dir, f'{tag}_aggregated.done'), 'w').close()
        return

    agg_parts = {}

    # 1. State-pair summary (median/HPD across sims in this batch)
    _, sp_sum = aggregate_state_pairs(
        trans_df, county_state, tmrca_offsets, tx_to_npi_days)
    agg_parts['state_pairs'] = sp_sum

    # 2. Weekly temporal profiles summary
    _, wp_sum = aggregate_weekly_profiles(trans_df, county_state)
    agg_parts['weekly_profiles'] = wp_sum

    # 3. Chain depth — summarize per scenario/state (median/HPD), not per sim
    cd = compute_chain_depth(trans_df, county_state, seed_county_default)
    if len(cd) > 0:
        cd_sum = cd.groupby(['scenario', 'state']).agg(
            chain_depth_median=('chain_depth', 'median'),
            chain_depth_q025=('chain_depth', lambda x: np.percentile(x, 2.5)),
            chain_depth_q975=('chain_depth', lambda x: np.percentile(x, 97.5)),
            first_intro_day_median=('first_intro_day', 'median'),
            first_intro_day_q025=('first_intro_day', lambda x: np.percentile(x, 2.5)),
            first_intro_day_q975=('first_intro_day', lambda x: np.percentile(x, 97.5)),
            most_common_source=('intro_source_state', lambda x: x.mode().iloc[0] if len(x) > 0 else ''),
            n_sims=('chain_depth', 'count'),
        ).reset_index()
        agg_parts['chain_depth'] = cd_sum

    # 4. Consequential transmissions — summarize per scenario
    if county_df is not None and len(county_df) > 0:
        cq_persim, _ = classify_consequential(
            trans_df, county_df, county_state)
        if len(cq_persim) > 0:
            cq_sum = cq_persim.groupby('scenario').agg(
                median_n_interstate=('n_interstate', 'median'),
                median_n_consequential=('n_consequential', 'median'),
                median_frac_consequential=('frac_consequential', 'median'),
                q025_frac=('frac_consequential', lambda x: np.percentile(x, 2.5)),
                q975_frac=('frac_consequential', lambda x: np.percentile(x, 97.5)),
                n_sims=('n_interstate', 'count'),
            ).reset_index()
            agg_parts['consequential'] = cq_sum

    # 5. Cumulative county profile — summarize per scenario/week
    if county_df is not None and len(county_df) > 0:
        cum = _compute_cumulative_profile(county_df)
        if len(cum) > 0:
            cum_sum = cum.groupby(['scenario', 'week']).agg(
                cumulative_median=('cumulative', 'median'),
                cumulative_q025=('cumulative', lambda x: np.percentile(x, 2.5)),
                cumulative_q25=('cumulative', lambda x: np.percentile(x, 25)),
                cumulative_q75=('cumulative', lambda x: np.percentile(x, 75)),
                cumulative_q975=('cumulative', lambda x: np.percentile(x, 97.5)),
                n_sims=('cumulative', 'count'),
            ).reset_index()
            agg_parts['cumulative'] = cum_sum

    # Save per-batch aggregate CSVs (small, fixed size per batch)
    for part_name, part_df in agg_parts.items():
        part_df.to_csv(
            os.path.join(batch_dir, f'{tag}_agg_{part_name}.csv'),
            index=False)

    # Delete raw files
    os.remove(trans_path)
    if county_df is not None:
        os.remove(county_path)

    # Write marker
    open(os.path.join(batch_dir, f'{tag}_aggregated.done'), 'w').close()


def _compute_cumulative_profile(county_df):
    """Compute weekly cumulative infected counties per sim."""
    df = county_df.copy()
    df['week'] = df['infection_day'] // 7

    per_sim_week = df.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx', 'week']
    ).size().reset_index(name='new_counties')

    per_sim_week = per_sim_week.sort_values(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx', 'week'])
    per_sim_week['cumulative'] = per_sim_week.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx']
    )['new_counties'].cumsum()

    return per_sim_week



##### Scenario expansion #####

def expand_scenarios(config, groups=None, scenario_filter=None,
                     all_county_fips=None):
    """Expand config scenario definitions into a flat list."""
    all_scenarios = []
    cfg_scenarios = config.get('scenarios', {})
    model_seed_county = config['model']['seed_county']

    if groups:
        target_groups = groups
    else:
        target_groups = list(cfg_scenarios.keys())
        if 'alternative_origins' in config:
            target_groups.append('alternative_origins')

    for group_name in target_groups:
        if group_name == 'all_county_origins':
            aco_cfg = config['all_county_origins']
            if all_county_fips is None:
                print("ERROR: all_county_origins requires county data "
                      "(pass all_county_fips)", flush=True)
                continue
            for fips in sorted(all_county_fips):
                st = county_to_state(fips)
                if st is None:
                    continue
                for npi_cond in aco_cfg['npi_conditions']:
                    sc_name = f'county_{fips}_{npi_cond}'
                    if npi_cond == 'npi':
                        npis = [{'target': 'interstate',
                                 'factor': 'posterior',
                                 'delay_post_tx': 35}]
                    else:
                        npis = []
                    all_scenarios.append({
                        'name': sc_name,
                        'group': 'all_county_origins',
                        'npis': npis,
                        'seed_county': fips,
                        'seed_state': st,
                        'connectivity_tier': None,
                        'npi_condition': npi_cond,
                        'description': f'County {fips} ({st}) origin, {npi_cond}',
                    })
            continue

        elif group_name == 'alternative_origins':
            ao_cfg = config['alternative_origins']
            for st in ao_cfg['seed_states']:
                for tier in ao_cfg['connectivity_tiers']:
                    for npi_cond in ao_cfg['npi_conditions']:
                        sc_name = f'{st}_{tier}_{npi_cond}'
                        if npi_cond == 'npi':
                            npis = [{'target': 'interstate',
                                     'factor': 'posterior',
                                     'delay_post_tx': 35}]
                        else:
                            npis = []

                        all_scenarios.append({
                            'name': sc_name,
                            'group': 'alternative_origins',
                            'npis': npis,
                            'seed_county': None,
                            'seed_state': st,
                            'connectivity_tier': tier,
                            'npi_condition': npi_cond,
                            'description': f'{st} origin, {tier} connectivity, {npi_cond}',
                        })
            continue

        if group_name not in cfg_scenarios:
            print(f"WARNING: unknown scenario group '{group_name}', skipping",
                  flush=True)
            continue

        group = cfg_scenarios[group_name]
        for sc_name, sc_def in group.items():
            all_scenarios.append({
                'name': sc_name,
                'group': group_name,
                'npis': sc_def.get('npis', []),
                'seed_county': sc_def.get('seed_county', model_seed_county),
                'seed_state': None,
                'connectivity_tier': None,
                'npi_condition': None,
                'description': sc_def.get('description', ''),
            })

    if scenario_filter:
        all_scenarios = [s for s in all_scenarios if s['name'] in scenario_filter]

    return all_scenarios


def get_npi_sim_day(scenario, tmrca_offset, config):
    """Get the earliest NPI activation sim day for classify_transmissions."""
    earliest = None
    for npi_def in scenario['npis']:
        if 'delay_post_tx' in npi_def:
            day = tmrca_offset + npi_def['delay_post_tx']
        elif 'delay_post_tmrca' in npi_def:
            day = npi_def['delay_post_tmrca']
        else:
            day = 0
        if earliest is None or day < earliest:
            earliest = day
    return earliest


##### Aggregation of results CSVs (no transmission logs needed) #####

def aggregate_results(output_dir, n_batches, scenarios):
    """Merge all batch results into final scenario-level summaries."""
    batch_dir = os.path.join(output_dir, 'batches')
    all_summaries = []

    for sc in scenarios:
        sc_name = sc['name']
        sim_dfs = []
        state_dfs = []
        for batch_idx in range(n_batches):
            tag = f'batch_{batch_idx:03d}_{sc_name}'
            sim_path = os.path.join(batch_dir, f'{tag}.csv')
            state_path = os.path.join(batch_dir, f'{tag}_states.csv')
            if os.path.exists(sim_path):
                sim_dfs.append(pd.read_csv(sim_path))
            if os.path.exists(state_path):
                state_dfs.append(pd.read_csv(state_path))

        if not sim_dfs:
            continue

        df = pd.concat(sim_dfs, ignore_index=True)
        df.to_csv(os.path.join(output_dir, f'{sc_name}_results.csv'),
                  index=False)

        if state_dfs:
            df_st = pd.concat(state_dfs, ignore_index=True)
            df_st.to_csv(os.path.join(output_dir,
                                      f'{sc_name}_state_detail.csv'),
                         index=False)

        n_dieout = (df['n_states'] == 1).sum()
        pct_dieout = n_dieout / len(df) * 100

        summary = {
            'scenario': sc_name,
            'group': sc['group'],
            'n_sims': len(df),
            'n_dieout': n_dieout,
            'pct_dieout': round(pct_dieout, 1),
            'median_states': df['n_states'].median(),
            'q025_states': df['n_states'].quantile(0.025),
            'q975_states': df['n_states'].quantile(0.975),
            'median_counties': df['n_counties'].median(),
            'q025_counties': df['n_counties'].quantile(0.025),
            'q975_counties': df['n_counties'].quantile(0.975),
            'median_interstate_pre': df['pre_interstate'].median(),
            'median_interstate_post': df['post_interstate'].median(),
            'median_intrastate_pre': df['pre_intrastate'].median(),
            'median_intrastate_post': df['post_intrastate'].median(),
        }

        if sc['group'] in ('alternative_origins', 'all_county_origins'):
            summary['seed_state'] = sc['seed_state']
            summary['connectivity_tier'] = sc['connectivity_tier']
            summary['npi_condition'] = sc['npi_condition']
            if sc['group'] == 'all_county_origins':
                summary['seed_county'] = sc['seed_county']
            for t in [1, 2, 3, 5, 10, 15, 20, 30]:
                summary[f'pct_leq_{t}_states'] = round(
                    (df['n_states'] <= t).sum() / len(df) * 100, 2)

        all_summaries.append(summary)

        print(f"  {sc_name} (n={len(df)}, dieout={pct_dieout:.1f}%): "
              f"states={df['n_states'].median():.0f} "
              f"[{df['n_states'].quantile(0.025):.0f}, "
              f"{df['n_states'].quantile(0.975):.0f}], "
              f"counties={df['n_counties'].median():.0f} "
              f"[{df['n_counties'].quantile(0.025):.0f}, "
              f"{df['n_counties'].quantile(0.975):.0f}]")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(output_dir, 'scenario_summary.csv'),
                      index=False)
    print(f"\nSummary saved to {output_dir}/scenario_summary.csv")


##### Main simulation loop #####

def main():
    parser = argparse.ArgumentParser(
        description='Run downstream simulation scenarios')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--abc_results', type=str,
                        default='outputs/abc_final/abc_accepted.csv')
    parser.add_argument('--groups', type=str, default=None,
                        help='Comma-separated scenario groups')
    parser.add_argument('--scenarios', type=str, default=None,
                        help='Comma-separated scenario names within groups')
    parser.add_argument('--num_networks', type=int, default=None)
    parser.add_argument('--num_sims', type=int, default=None)
    parser.add_argument('--num_days', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/scenarios')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--keep_raw', action='store_true',
                        help='Keep raw transmission logs (do not aggregate and delete)')
    parser.add_argument('--skip_state_detail', action='store_true',
                        help='Skip writing per-state detail rows (saves disk for large runs)')
    args = parser.parse_args()

    config = load_config(args.config)
    sim_cfg = config['simulation']

    groups = args.groups.split(',') if args.groups else None

    aco_num_nets = (config.get('all_county_origins', {}).get('num_networks')
                    if groups and 'all_county_origins' in groups else None)
    num_networks = args.num_networks or aco_num_nets or sim_cfg['num_networks']
    num_sims = args.num_sims or sim_cfg['num_sims']
    num_days = args.num_days or config['model']['num_days']
    batch_size = args.batch_size or sim_cfg['batch_size']
    master_seed = args.seed if args.seed is not None else sim_cfg['seed']
    model_seed_county = config['model']['seed_county']
    tx_to_npi_days = config['timeline']['tx_to_npi_days']

    TX_DETECTION_DATE = date.fromisoformat(
        config['timeline']['tx_detection_date'])

    # ── Expand scenarios ──
    scenario_filter = (set(args.scenarios.split(','))
                       if args.scenarios else None)

    all_county_fips = None
    if groups and 'all_county_origins' in groups:
        Sz_early, _ = load_county_data(config)
        all_county_fips = list(Sz_early.keys())
        del Sz_early

    scenarios = expand_scenarios(config, groups=groups,
                                scenario_filter=scenario_filter,
                                all_county_fips=all_county_fips)

    if not scenarios:
        print("ERROR: no scenarios selected.", flush=True)
        sys.exit(1)

    scenario_names = [s['name'] for s in scenarios]
    has_alt_origins = any(s['group'] == 'alternative_origins' for s in scenarios)

    # ── Load data ──
    print("Loading county data...", flush=True)
    Sz, Pz = load_county_data(config)
    county_state = build_county_state_map(Sz)

    all_network_paths = sorted([
        os.path.join(resolve_path(config, 'network_dir'), f)
        for f in os.listdir(resolve_path(config, 'network_dir'))
        if f.startswith('dairy_network_') and f.endswith('.network')
    ])

    # ── Load ABC posterior ──
    print(f"Loading ABC posterior from {args.abc_results}...", flush=True)
    abc_df = pd.read_csv(args.abc_results)
    n_param_sets = len(abc_df)
    print(f"  {n_param_sets} accepted parameter sets", flush=True)

    # Build tmrca_offsets lookup for aggregation
    tmrca_offsets = {}
    for idx, row in abc_df.iterrows():
        tmrca_offsets[idx] = int(row['tmrca_offset'])

    # ── Select networks ──
    rng = np.random.default_rng(master_seed)
    random.seed(master_seed)

    net_indices = rng.choice(len(all_network_paths),
                             size=min(num_networks, len(all_network_paths)),
                             replace=False)
    selected_network_paths = [all_network_paths[i] for i in net_indices]
    n_nets = len(selected_network_paths)
    n_scenarios = len(scenarios)
    n_batches = (n_nets + batch_size - 1) // batch_size
    total_sims_target = n_param_sets * n_nets * num_sims * n_scenarios

    print(f"\nPlan: {n_param_sets} params × {n_nets} networks × {num_sims} sims "
          f"× {n_scenarios} scenarios = {total_sims_target:,} simulations",
          flush=True)
    print(f"Batching: {n_batches} batches of up to {batch_size}", flush=True)
    print(f"Groups: {list(set(s['group'] for s in scenarios))}", flush=True)
    if not args.keep_raw:
        print("Aggregating and deleting raw logs after each batch.", flush=True)

    # ── Build connectivity pools if needed ──
    pools = None
    if has_alt_origins:
        print("\nBuilding connectivity pools...", flush=True)
        ao_cfg = config['alternative_origins']
        pools, county_degree = build_connectivity_pools(
            Sz, config,
            n_sample_nets=ao_cfg.get('n_sample_nets_for_pools', 20))

    # ── Save provenance ──
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(args.abc_results,
                 os.path.join(args.output_dir, 'abc_accepted.csv'))

    metadata = {
        'command': ' '.join(sys.argv),
        'seed': master_seed,
        'num_networks': n_nets,
        'num_sims': num_sims,
        'num_days': num_days,
        'batch_size': batch_size,
        'n_param_sets': n_param_sets,
        'scenario_names': scenario_names,
        'groups': list(set(s['group'] for s in scenarios)),
        'network_files': [os.path.basename(p) for p in selected_network_paths],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    if pools:
        pool_info = {}
        for st in config['alternative_origins']['seed_states']:
            if st in pools:
                pool_info[st] = {
                    'high': pools[st]['high'],
                    'low': pools[st]['low'],
                    'high_degrees': [county_degree.get(c, 0)
                                     for c in pools[st]['high']],
                    'low_degrees': [county_degree.get(c, 0)
                                    for c in pools[st]['low']],
                }
        with open(os.path.join(args.output_dir,
                               'connectivity_pools.json'), 'w') as f:
            json.dump(pool_info, f, indent=2)

    print("Provenance saved.", flush=True)

    # ── Check for resume ──
    completed_pairs = get_completed_scenario_batches(args.output_dir,
                                                     scenario_names)
    if completed_pairs:
        print(f"Found {len(completed_pairs)} completed (batch, scenario) pairs "
              f"— will skip.", flush=True)

    total_sims = 0
    t0 = time.time()

        # Main loop
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_nets)
        batch_paths = selected_network_paths[batch_start:batch_end]
        batch_n = len(batch_paths)

        scenarios_to_run = [sc for sc in scenarios
                           if (batch_idx, sc['name']) not in completed_pairs]
        if not scenarios_to_run:
            total_sims += batch_n * n_param_sets * num_sims * n_scenarios
            print(f"\n── Batch {batch_idx+1}/{n_batches}: "
                  f"all scenarios done, skipping ──", flush=True)
            continue

        print(f"\n── Batch {batch_idx+1}/{n_batches}: "
              f"networks {batch_start+1}-{batch_end} "
              f"({len(scenarios_to_run)}/{n_scenarios} scenarios) ──",
              flush=True)

        loaded_adj_indices = []
        for i, net_path in enumerate(batch_paths):
            daily_edges = load_network(net_path)
            loaded_adj_indices.append(build_adj_index(daily_edges))
            if (i + 1) % 10 == 0 or i + 1 == batch_n:
                print(f"  Loaded {i+1}/{batch_n}", flush=True)

        for sc in scenarios_to_run:
            sc_name = sc['name']
            if (batch_idx, sc_name) in completed_pairs:
                total_sims += batch_n * n_param_sets * num_sims
                continue

            batch_results = []
            batch_state_rows = []
            raw_writer = ScenarioBatchWriter(args.output_dir, batch_idx,
                                             sc_name)

            for pidx, row in abc_df.iterrows():
                beta = row['beta']
                alpha = row['alpha']
                gamma = row['gamma']
                mu_c = int(row['mu_c'])
                npi_factor_posterior = row['npi_factor']
                tmrca_offset = int(row['tmrca_offset'])

                mu_C = compute_recovery_periods(Sz, Pz, alpha, gamma, mu_c)
                tmrca_doy = get_tmrca_doy(
                    tmrca_offset, config['timeline']['tx_detection_doy'])

                npi_layers = build_npi_layers(
                    sc['npis'], tmrca_offset, county_state,
                    npi_factor_posterior, config)

                npi_sim_day = get_npi_sim_day(sc, tmrca_offset, config)

                for local_net_idx, adj_index in enumerate(loaded_adj_indices):
                    global_net_idx = batch_start + local_net_idx

                    for sim_i in range(num_sims):
                        total_sims += 1

                        sim_seed = make_sim_seed(master_seed, pidx,
                                                  global_net_idx, sim_i,
                                                  sc_name)
                        sim_rng = random.Random(sim_seed)

                        if sc['group'] == 'alternative_origins':
                            pool = pools[sc['seed_state']][sc['connectivity_tier']]
                            if not pool:
                                continue
                            seed_county = sim_rng.choice(pool)
                        else:
                            seed_county = sc['seed_county']

                        inf_times, log = simulate_sir(
                            adj_index, Sz, mu_C, beta, [seed_county],
                            num_days, npi_layers, tmrca_doy, rng=sim_rng)

                        raw_writer.write_sim(
                            pidx, global_net_idx, sim_i, seed_county,
                            log, inf_times, mu_C)

                        state_arrivals = extract_state_arrivals(inf_times)
                        trans_class = classify_transmissions(
                            log, npi_sim_day, county_state)
                        s_metrics = compute_state_metrics(
                            inf_times, log, county_state)

                        tmrca_date = (TX_DETECTION_DATE
                                      - timedelta(days=tmrca_offset))

                        # Cumulative county counts at weekly checkpoints
                        checkpoints = {}
                        for wk in [4, 8, 12, 16, 20, 24, 32, 40, 52, 64, 78, 92, 104]:
                            day = wk * 7
                            checkpoints[f'counties_w{wk}'] = sum(
                                1 for d in inf_times.values() if d <= day)

                        result_row = {
                            'param_idx': pidx,
                            'net_idx': global_net_idx,
                            'sim_idx': sim_i,
                            'scenario': sc_name,
                            'seed_county': seed_county,
                            'tmrca_offset': tmrca_offset,
                            'start_date': str(tmrca_date),
                            'n_states': len(state_arrivals),
                            'n_counties': len(inf_times),
                            **trans_class,
                            **checkpoints,
                        }

                        if sc['group'] in ('alternative_origins',
                                              'all_county_origins'):
                            result_row['seed_state'] = sc['seed_state']
                            result_row['connectivity_tier'] = sc['connectivity_tier']
                            result_row['npi_condition'] = sc['npi_condition']

                        batch_results.append(result_row)

                        if not args.skip_state_detail:
                            for st in s_metrics['state_arrivals']:
                                arr_day = s_metrics['state_arrivals'][st]
                                state_row = {
                                    'param_idx': pidx,
                                    'net_idx': global_net_idx,
                                    'sim_idx': sim_i,
                                    'scenario': sc_name,
                                    'state': st,
                                    'arrival_day': arr_day,
                                    'arrival_date': str(
                                        tmrca_date + timedelta(days=arr_day)),
                                    'n_introductions':
                                        s_metrics['state_introductions'].get(st, 0),
                                    'n_intrastate':
                                        s_metrics['state_intrastate'].get(st, 0),
                                    'n_counties':
                                        s_metrics['state_infected_counties'].get(st, 0),
                                }
                                if sc['group'] in ('alternative_origins',
                                                    'all_county_origins'):
                                    state_row['seed_state'] = sc['seed_state']
                                    state_row['seed_county'] = seed_county
                                batch_state_rows.append(state_row)

                raw_writer.flush()

            raw_writer.close()

            save_batch_results(args.output_dir, batch_idx, sc_name,
                              batch_results)
            if not args.skip_state_detail:
                save_batch_state_detail(args.output_dir, batch_idx, sc_name,
                                       batch_state_rows)

            # Aggregate and cleanup raw files
            if not args.keep_raw:
                aggregate_and_cleanup_batch(
                    args.output_dir, batch_idx, sc_name,
                    county_state, tmrca_offsets, tx_to_npi_days,
                    model_seed_county)

            elapsed = time.time() - t0
            rate = total_sims / elapsed if elapsed > 0 else 0
            remaining = ((total_sims_target - total_sims) / rate / 60
                        if rate > 0 else 0)
            print(f"  {sc_name}: batch {batch_idx+1} done "
                  f"({total_sims:,}/{total_sims_target:,}, "
                  f"{rate:.1f} sims/s, ~{remaining:.0f} min left)",
                  flush=True)

        del loaded_adj_indices
        print(f"  Batch {batch_idx+1}/{n_batches} complete.", flush=True)

    # ── Aggregate results CSVs ──
    print("\n── Aggregating all batch results ──", flush=True)
    aggregate_results(args.output_dir, n_batches, scenarios)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/3600:.1f} hours")


if __name__ == '__main__':
    main()
