"""
ABC-SMC calibration for county-level H5N1 SIR simulation.

Fits epidemic parameters (beta, alpha, gamma, mu_c, npi_factor, tmrca_offset)
to observed state-level outbreak data using sequential Monte Carlo.

Distance function settings are configurable for sensitivity analysis.

Usage:
    # Default (intensity mode, baseline settings)
    python abc_calibration.py --output_dir outputs/abc_final

    # Sensitivity: vary missing penalty
    python abc_calibration.py --missing_penalty 1.0 --output_dir outputs/abc_sensitivity/mp1

    # Sensitivity: disable intensity component
    python abc_calibration.py --include_intensity false --output_dir outputs/abc_sensitivity/no_intensity

    # Resume from previous round
    python abc_calibration.py --resume_from outputs/abc_final/abc_accepted.csv --output_dir outputs/abc_final
"""

import os
import sys
import math
import random
import argparse
import json
import time
import shutil

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sir_model import (
    load_config, resolve_path, load_county_data, load_network, build_adj_index,
    compute_recovery_periods, simulate_sir, build_npi_layers,
    extract_state_arrivals, extract_state_county_counts,
    build_county_state_map, get_tmrca_doy, NpiLayer,
)


##### Distance functions #####

def compute_distance(sim_state_days, tmrca_offset, observed_days,
                     observed_counts=None, sim_state_counts=None,
                     mode='intensity', missing_penalty=2.0,
                     early_late_split_days=30, include_intensity=True):
    """Compute distance between simulated and observed outbreak patterns.

    Returns:
        (distance, n_matched, rank_corr)
    """
    obs_sim_days = {st: d + tmrca_offset for st, d in observed_days.items()}

    obs_states = set(obs_sim_days.keys()) - {'TX'}
    sim_states = set(sim_state_days.keys()) - {'TX'}

    matched = obs_states & sim_states
    if len(matched) < 3:
        return float('inf'), len(matched), 0.0

    early_states = {st for st in matched
                    if observed_days[st] <= early_late_split_days}
    late_states = matched - early_states

    # Early penalty: asymmetric, normalized by observed day
    early_penalty = 0.0
    for st in early_states:
        obs_day = obs_sim_days[st]
        sim_day = sim_state_days[st]
        if sim_day > obs_day:
            early_penalty += (sim_day - obs_day) / obs_day

    # Late penalty: log-scale RMSE
    late_log_rmse = 0.0
    if late_states:
        obs_late = np.array([obs_sim_days[st] for st in sorted(late_states)],
                            dtype=float)
        sim_late = np.array([sim_state_days[st] for st in sorted(late_states)],
                            dtype=float)
        late_log_rmse = np.sqrt(np.mean(
            (np.log1p(obs_late) - np.log1p(sim_late)) ** 2
        ))

    # Rank correlation across all matched states
    obs_vals = [obs_sim_days[st] for st in sorted(matched)]
    sim_vals = [sim_state_days[st] for st in sorted(matched)]
    rho, _ = spearmanr(obs_vals, sim_vals)

    # Missing observed states penalty
    n_missing = len(obs_states - sim_states)
    miss_pen = n_missing * missing_penalty

    rank_dist = 1.0 - max(rho, 0.0)
    distance = rank_dist + late_log_rmse + early_penalty + miss_pen

    # Intensity component
    if mode == 'intensity' and include_intensity and sim_state_counts and observed_counts:
        count_matched = set(observed_counts.keys()) & set(sim_state_counts.keys())
        if len(count_matched) >= 3:
            obs_c = [observed_counts[st] for st in sorted(count_matched)]
            sim_c = [sim_state_counts[st] for st in sorted(count_matched)]
            rho_intensity, _ = spearmanr(obs_c, sim_c)
            distance += 1.0 - max(rho_intensity, 0.0)

    return distance, len(matched), rho


##### Proposal runner #####

def run_single_proposal(proposal_id, params, preloaded_networks, Sz, Pz,
                        num_sims, num_days, seed_county, county_state,
                        config, observed_days, observed_counts,
                        mode, missing_penalty, early_late_split_days,
                        include_intensity):
    """Run simulations for one parameter proposal across preloaded networks."""
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    mu_c = int(params['mu_c'])
    npi_factor = params['npi_factor']
    tmrca_offset = int(params['tmrca_offset'])

    tx_to_npi = config['timeline']['tx_to_npi_days']
    npi_sim_day = tmrca_offset + tx_to_npi
    tmrca_doy = get_tmrca_doy(tmrca_offset,
                               config['timeline']['tx_detection_doy'])
    mu_C = compute_recovery_periods(Sz, Pz, alpha, gamma, mu_c)

    # ABC calibration uses a single interstate NPI layer at the federal order
    npi_layers = [NpiLayer(
        target='interstate',
        factor=npi_factor,
        activation_day=npi_sim_day,
        county_state=county_state,
    )]

    all_state_arrivals = []
    all_state_counts = []

    for adj_index in preloaded_networks:
        for _ in range(num_sims):
            inf_times, _ = simulate_sir(
                adj_index, Sz, mu_C, beta, [seed_county], num_days,
                npi_layers, tmrca_doy
            )
            all_state_arrivals.append(extract_state_arrivals(inf_times))
            all_state_counts.append(extract_state_county_counts(inf_times))

    # Median arrival per state
    state_collections = {}
    for sa in all_state_arrivals:
        for st, day in sa.items():
            state_collections.setdefault(st, []).append(day)
    median_arrivals = {st: np.median(ds) for st, ds in state_collections.items()}

    # Median county counts per state
    count_collections = {}
    for sc in all_state_counts:
        for st, c in sc.items():
            count_collections.setdefault(st, []).append(c)
    median_counts = {st: np.median(cs) for st, cs in count_collections.items()}

    distance, n_matched, rank_corr = compute_distance(
        median_arrivals, tmrca_offset, observed_days,
        observed_counts=observed_counts,
        sim_state_counts=median_counts,
        mode=mode,
        missing_penalty=missing_penalty,
        early_late_split_days=early_late_split_days,
        include_intensity=include_intensity,
    )

    return {
        'proposal_id': proposal_id,
        'params': params,
        'distance': distance,
        'n_matched': n_matched,
        'rank_corr': rank_corr,
        'median_arrivals': median_arrivals,
        'median_counts': median_counts,
        'n_sims': len(all_state_arrivals),
    }


##### Prior sampling and perturbation #####

def sample_params(prior_bounds, int_params, rng):
    params = {}
    for pname, (lo, hi) in prior_bounds.items():
        if pname in int_params:
            params[pname] = int(rng.integers(lo, hi + 1))
        else:
            params[pname] = rng.uniform(lo, hi)
    return params


def perturb_params(particle, kernel_scales, prior_bounds, int_params, rng):
    new_params = {}
    for pname, value in particle.items():
        lo, hi = prior_bounds[pname]
        scale = kernel_scales[pname]
        new_val = value + rng.normal(0, scale)
        while new_val < lo or new_val > hi:
            if new_val < lo:
                new_val = lo + (lo - new_val)
            if new_val > hi:
                new_val = hi - (new_val - hi)
        if pname in int_params:
            new_val = int(round(new_val))
            new_val = max(lo, min(hi, new_val))
        new_params[pname] = new_val
    return new_params


def compute_kernel_scales(accepted_params_list, prior_bounds):
    scales = {}
    for pname in prior_bounds:
        vals = [p[pname] for p in accepted_params_list]
        scales[pname] = 2.0 * np.std(vals)
        lo, hi = prior_bounds[pname]
        scales[pname] = max(scales[pname], 0.01 * (hi - lo))
    return scales


##### Reporting #####

def print_posterior(accepted, param_names):
    posterior_summary = {}
    for pname in param_names:
        vals = [r['params'][pname] for r in accepted]
        posterior_summary[pname] = {
            'median': float(np.median(vals)),
            'std': float(np.std(vals)),
            'q025': float(np.percentile(vals, 2.5)),
            'q975': float(np.percentile(vals, 97.5)),
        }
        print(f"  {pname:14s}: median={np.median(vals):.4f}  "
              f"95% HPD=[{np.percentile(vals, 2.5):.4f}, "
              f"{np.percentile(vals, 97.5):.4f}]")
    return posterior_summary


def print_best_proposal(best, param_names, observed_days):
    best_offset = int(best['params']['tmrca_offset'])
    print(f"\n── Best proposal (dist={best['distance']:.3f}, "
          f"rho={best['rank_corr']:.3f}, tmrca_offset={best_offset}) ──")
    for pname in param_names:
        print(f"  {pname}: {best['params'][pname]:.4f}")

    obs_shifted = {st: d + best_offset for st, d in observed_days.items()}
    print(f"\n  State arrivals (sim days | observed):")
    for st in sorted(best['median_arrivals'],
                     key=lambda s: best['median_arrivals'][s]):
        sim_d = best['median_arrivals'][st]
        obs_d = obs_shifted.get(st, '—')
        obs_det = observed_days.get(st, '')
        marker = '*' if st in observed_days else ' '
        print(f"  {marker} {st}: sim={sim_d:.0f}  obs={obs_d}"
              + (f"  (det day {obs_det})" if obs_det != '' else ''))


def save_results(results, accepted, threshold, posterior_summary, mode,
                 output_dir, observed_days, observed_counts):
    os.makedirs(output_dir, exist_ok=True)

    param_names = list(results[0]['params'].keys())
    rows = []
    for r in results:
        row = {'proposal_id': r['proposal_id'], 'distance': r['distance'],
               'n_matched': r['n_matched'], 'rank_corr': r['rank_corr']}
        row.update(r['params'])
        rows.append(row)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, 'abc_all_proposals.csv'), index=False)

    df_accepted = df_all[df_all['distance'] <= threshold + 1e-10]
    df_accepted.to_csv(os.path.join(output_dir, 'abc_accepted.csv'), index=False)

    with open(os.path.join(output_dir, 'abc_posterior_summary.json'), 'w') as f:
        json.dump({'mode': mode, 'parameters': posterior_summary}, f, indent=2)

    best = results[0]
    best_offset = int(best['params']['tmrca_offset'])
    obs_shifted = {st: d + best_offset for st, d in observed_days.items()}
    best_rows = []
    for st in sorted(best['median_arrivals']):
        best_rows.append({
            'state': st,
            'sim_day': best['median_arrivals'][st],
            'obs_detection_day': observed_days.get(st, np.nan),
            'obs_day_since_tmrca': obs_shifted.get(st, np.nan),
            'sim_county_count': best['median_counts'].get(st, 0),
            'obs_outbreak_count': observed_counts.get(st, np.nan),
        })
    pd.DataFrame(best_rows).to_csv(
        os.path.join(output_dir, 'abc_best_arrivals.csv'), index=False)


##### SMC round #####

def run_smc_round(round_num, num_proposals, prev_accepted, network_pool,
                  Sz, Pz, county_state, num_networks, num_sims, num_days,
                  seed_county, config, observed_days, observed_counts,
                  mode, missing_penalty, early_late_split_days,
                  include_intensity, acceptance_quantile, prior_bounds,
                  int_params, rng):
    """Run one round of ABC-SMC."""
    pool_size = len(network_pool)
    param_names = list(prior_bounds.keys())

    kernel_scales = None
    if prev_accepted is not None:
        prev_params = [r['params'] for r in prev_accepted]
        kernel_scales = compute_kernel_scales(prev_params, prior_bounds)
        print(f"\n  Kernel scales: " +
              ", ".join(f"{p}={kernel_scales[p]:.4f}" for p in param_names))
        print(f"  Previous threshold: {prev_accepted[-1]['distance']:.3f}")
        sys.stdout.flush()

    results = []
    t0 = time.time()
    best_dist = float('inf')

    for i in range(num_proposals):
        if prev_accepted is None:
            params = sample_params(prior_bounds, int_params, rng)
        else:
            idx = rng.integers(0, len(prev_accepted))
            base_params = prev_accepted[idx]['params']
            params = perturb_params(base_params, kernel_scales,
                                    prior_bounds, int_params, rng)

        net_idx = rng.choice(pool_size, size=num_networks, replace=False)
        proposal_networks = [network_pool[j] for j in net_idx]

        result = run_single_proposal(
            i, params, proposal_networks, Sz, Pz,
            num_sims, num_days, seed_county, county_state,
            config, observed_days, observed_counts,
            mode, missing_penalty, early_late_split_days, include_intensity,
        )
        results.append(result)
        if result['distance'] < best_dist:
            best_dist = result['distance']

        done = i + 1
        if done % 25 == 0 or done == 1 or done == num_proposals:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (num_proposals - done) / rate if rate > 0 else 0
            print(f"  [{done}/{num_proposals}] "
                  f"dist={result['distance']:.3f} "
                  f"rho={result['rank_corr']:.3f} "
                  f"best={best_dist:.3f} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")
            sys.stdout.flush()

    total_time = time.time() - t0
    print(f"  Round {round_num} completed in {total_time:.1f}s "
          f"({total_time/num_proposals:.1f}s/proposal)")

    results.sort(key=lambda r: r['distance'])
    n_accept = max(1, int(len(results) * acceptance_quantile))
    accepted = results[:n_accept]
    threshold = accepted[-1]['distance']

    print(f"  Accepted {n_accept}/{len(results)} "
          f"(threshold: {threshold:.3f})")

    return results, accepted, threshold


##### Main #####

def main():
    parser = argparse.ArgumentParser(
        description='ABC-SMC calibration for H5N1 SIR simulation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: config.yaml in script dir)')
    parser.add_argument('--mode', type=str, default=None,
                        help='Override: timing or intensity')
    parser.add_argument('--num_rounds', type=int, default=None)
    parser.add_argument('--num_proposals', type=int, default=None)
    parser.add_argument('--num_networks', type=int, default=None,
                        help='Networks per proposal')
    parser.add_argument('--num_sims', type=int, default=None,
                        help='Sims per network')
    parser.add_argument('--num_days', type=int, default=None)
    parser.add_argument('--network_pool_size', type=int, default=None)
    parser.add_argument('--acceptance_quantile', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/abc')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to previous abc_accepted.csv to seed round 1')

    # Distance function overrides (for sensitivity analysis)
    parser.add_argument('--missing_penalty', type=float, default=None)
    parser.add_argument('--early_late_split_days', type=int, default=None)
    parser.add_argument('--include_intensity', type=str, default=None,
                        help='true/false')
    parser.add_argument('--trajectories_per_proposal', type=int, default=None,
                        help='Override sims_per_network × networks_per_proposal')

    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve settings: CLI overrides > config
    abc_cfg = config['abc']
    smc_cfg = abc_cfg['smc']
    dist_cfg = abc_cfg['distance']

    mode = args.mode or dist_cfg['mode']
    num_rounds = args.num_rounds or smc_cfg['num_rounds']
    num_proposals = args.num_proposals or smc_cfg['num_proposals']
    num_networks = args.num_networks or smc_cfg['networks_per_proposal']
    num_sims = args.num_sims or smc_cfg['sims_per_network']
    num_days = args.num_days or config['model']['num_days']
    pool_size = args.network_pool_size or smc_cfg['network_pool_size']
    acceptance_q = args.acceptance_quantile or smc_cfg['acceptance_quantile']
    seed = args.seed if args.seed is not None else config['model']['master_seed']
    seed_county = config['model']['seed_county']

    missing_penalty = (args.missing_penalty if args.missing_penalty is not None
                       else dist_cfg['missing_penalty'])
    early_late_split = (args.early_late_split_days
                        if args.early_late_split_days is not None
                        else dist_cfg['early_late_split_days'])
    include_intensity = dist_cfg['include_intensity']
    if args.include_intensity is not None:
        include_intensity = args.include_intensity.lower() == 'true'

    if args.trajectories_per_proposal is not None:
        # Override: e.g. 15 total = 5 nets × 3 sims or 3 nets × 5 sims
        num_sims = args.trajectories_per_proposal // num_networks
        if num_sims < 1:
            num_sims = 1
            num_networks = args.trajectories_per_proposal

    prior_bounds = {k: tuple(v) for k, v in abc_cfg['prior_bounds'].items()}
    int_params = set(abc_cfg['integer_params'])
    param_names = list(prior_bounds.keys())

    observed_days = config['observed']['first_detection_days']
    observed_counts = config['observed']['outbreak_counts']

    print(f"ABC-SMC Calibration — mode: {mode}, {num_rounds} rounds")
    print(f"Distance: missing_penalty={missing_penalty}, "
          f"early_late_split={early_late_split}, "
          f"include_intensity={include_intensity}")
    print(f"Per round: {num_proposals} proposals, "
          f"{num_networks} nets × {num_sims} sims = "
          f"{num_networks * num_sims} trajectories/proposal")
    print(f"Observed states: {len(observed_days)} (no OR)")
    sys.stdout.flush()

    print("\nLoading county data...")
    Sz, Pz = load_county_data(config)
    county_state = build_county_state_map(Sz)
    print(f"  {len(Sz)} counties loaded")

    all_network_paths = sorted([
        os.path.join(resolve_path(config, 'network_dir'), f)
        for f in os.listdir(resolve_path(config, 'network_dir'))
        if f.startswith('dairy_network_') and f.endswith('.network')
    ])
    print(f"  {len(all_network_paths)} network realizations available")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    actual_pool_size = min(pool_size, len(all_network_paths))
    pool_indices = rng.choice(len(all_network_paths), size=actual_pool_size,
                              replace=False)
    print(f"\nPreloading {actual_pool_size} networks into memory...")
    sys.stdout.flush()

    network_pool = []
    t_load = time.time()
    for i, idx in enumerate(pool_indices):
        daily_edges = load_network(all_network_paths[idx])
        network_pool.append(build_adj_index(daily_edges))
        if (i + 1) % 10 == 0 or i + 1 == actual_pool_size:
            print(f"  {i+1}/{actual_pool_size} loaded "
                  f"({time.time()-t_load:.1f}s)")
            sys.stdout.flush()

    # Save run metadata
    os.makedirs(args.output_dir, exist_ok=True)
    run_meta = {
        'command': ' '.join(sys.argv),
        'seed': seed,
        'mode': mode,
        'num_rounds': num_rounds,
        'num_proposals': num_proposals,
        'networks_per_proposal': num_networks,
        'sims_per_network': num_sims,
        'network_pool_size': actual_pool_size,
        'num_days': num_days,
        'acceptance_quantile': acceptance_q,
        'missing_penalty': missing_penalty,
        'early_late_split_days': early_late_split,
        'include_intensity': include_intensity,
        'n_observed_states': len(observed_days),
        'observed_states': list(observed_days.keys()),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_meta, f, indent=2)

    # ── SMC Rounds ──
    prev_accepted = None
    start_round = 1

    if args.resume_from:
        print(f"\nResuming from {args.resume_from}...")
        prev_df = pd.read_csv(args.resume_from)
        prev_accepted = []
        for _, row in prev_df.iterrows():
            params = {p: row[p] for p in param_names}
            for ip in int_params:
                params[ip] = int(params[ip])
            prev_accepted.append({
                'params': params,
                'distance': row['distance'],
                'rank_corr': row['rank_corr'],
            })
        prev_accepted.sort(key=lambda r: r['distance'])
        print(f"  Loaded {len(prev_accepted)} particles "
              f"(threshold: {prev_accepted[-1]['distance']:.3f})")
        start_round = 2

    all_round_results = {}
    for round_num in range(start_round, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}/{num_rounds}"
              + (" (from prior)" if prev_accepted is None else " (SMC)"))
        print(f"{'='*60}")
        sys.stdout.flush()

        results, accepted, threshold = run_smc_round(
            round_num, num_proposals, prev_accepted, network_pool,
            Sz, Pz, county_state, num_networks, num_sims, num_days,
            seed_county, config, observed_days, observed_counts,
            mode, missing_penalty, early_late_split, include_intensity,
            acceptance_q, prior_bounds, int_params, rng,
        )

        print(f"\n── Round {round_num} Posterior ({mode} mode) ──")
        posterior_summary = print_posterior(accepted, param_names)
        print_best_proposal(results[0], param_names, observed_days)

        round_dir = os.path.join(args.output_dir, f'round_{round_num}')
        save_results(results, accepted, threshold, posterior_summary,
                     mode, round_dir, observed_days, observed_counts)
        print(f"\n  Round {round_num} results saved to {round_dir}/")
        sys.stdout.flush()

        all_round_results[round_num] = {
            'threshold': threshold,
            'best_distance': results[0]['distance'],
            'n_accepted': len(accepted),
        }
        prev_accepted = accepted

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  ABC-SMC COMPLETE — {num_rounds} rounds")
    print(f"{'='*60}")
    for rnd, info in all_round_results.items():
        print(f"  Round {rnd}: threshold={info['threshold']:.3f}, "
              f"best={info['best_distance']:.3f}, "
              f"accepted={info['n_accepted']}")

    final_round = max(all_round_results.keys())
    final_dir = os.path.join(args.output_dir, f'round_{final_round}')
    for fname in ['abc_accepted.csv', 'abc_all_proposals.csv',
                  'abc_posterior_summary.json', 'abc_best_arrivals.csv']:
        src = os.path.join(final_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    print(f"\nFinal results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
