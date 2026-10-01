"""
Server-side aggregation of transmission logs into compact summaries.

Run after downstream simulations complete. Reads batch transmission logs
and county_times, produces small summary CSVs for transfer back to laptop.

Outputs two levels of aggregation:
  - Per-simulation summaries (medium size, for detailed analysis)
  - Cross-simulation summaries (small, for quick plotting)

Usage:
    python aggregate.py --input_dir outputs/scenarios
    python aggregate.py --input_dir outputs/scenarios --scenarios npi_day35,no_npi
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

from sir_model import load_config, build_county_state_map


def add_state_columns(df, county_state):
    df['src_state'] = df['src_county'].map(county_state)
    df['dest_state'] = df['dest_county'].map(county_state)
    return df


##### 1. State-pair transmission matrices (per-sim → then summarized) #####

def aggregate_state_pairs(trans_df, county_state, tmrca_offsets, tx_to_npi_days):
    """Per-sim state-pair x era counts, then median/HPD across sims."""
    df = trans_df.copy()
    add_state_columns(df, county_state)
    df = df.dropna(subset=['src_state', 'dest_state'])

    offset = df['param_idx'].map(tmrca_offsets).fillna(100).astype(int)
    df['era'] = np.where(
        df['sim_day'] < offset, 'pre_detection',
        np.where(df['sim_day'] < offset + tx_to_npi_days,
                 'detection_to_order', 'post_order'))

    per_sim = df.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx',
         'src_state', 'dest_state', 'era']
    ).size().reset_index(name='n')

    summary = per_sim.groupby(
        ['scenario', 'src_state', 'dest_state', 'era']
    )['n'].agg(
        median='median',
        q025=lambda x: np.percentile(x, 2.5),
        q975=lambda x: np.percentile(x, 97.5),
        mean='mean',
        n_sims='count',
    ).reset_index()

    return per_sim, summary


##### 2. Weekly temporal profiles (per-sim → then summarized) #####

def aggregate_weekly_profiles(trans_df, county_state):
    df = trans_df.copy()
    add_state_columns(df, county_state)
    df['week'] = df['sim_day'] // 7
    df['is_interstate'] = (df['src_state'] != df['dest_state']).astype(int)
    df['is_intrastate'] = 1 - df['is_interstate']

    per_sim = df.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx', 'week']
    ).agg(
        interstate=('is_interstate', 'sum'),
        intrastate=('is_intrastate', 'sum'),
    ).reset_index()

    summary = per_sim.groupby(['scenario', 'week']).agg(
        interstate_median=('interstate', 'median'),
        interstate_q025=('interstate', lambda x: np.percentile(x, 2.5)),
        interstate_q975=('interstate', lambda x: np.percentile(x, 97.5)),
        intrastate_median=('intrastate', 'median'),
        intrastate_q025=('intrastate', lambda x: np.percentile(x, 2.5)),
        intrastate_q975=('intrastate', lambda x: np.percentile(x, 97.5)),
        n_sims=('interstate', 'count'),
    ).reset_index()

    return per_sim, summary


##### 3. Chain depth (per-sim only — already compact) #####

def compute_chain_depth(trans_df, county_state, seed_county_default):
    """Chain depth for each state's first interstate intro per sim."""
    rows = []
    sim_groups = trans_df.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx'], sort=False)

    for key, group in sim_groups:
        scenario, pidx, nidx, sidx = key
        group = group.sort_values('sim_day')

        seed_county = seed_county_default
        if 'seed_county' in group.columns:
            first_seed = group['seed_county'].iloc[0]
            if pd.notna(first_seed):
                seed_county = int(first_seed)

        county_depth = {seed_county: 0}
        state_first_intro = {}

        for t in group.itertuples(index=False):
            src = t.src_county
            dest = t.dest_county

            if dest not in county_depth:
                county_depth[dest] = county_depth.get(src, 0) + 1

            src_st = county_state.get(src)
            dest_st = county_state.get(dest)
            if src_st != dest_st and dest_st not in state_first_intro:
                state_first_intro[dest_st] = (
                    t.sim_day, county_depth[dest], src_st)

        for state, (day, depth, src_st) in state_first_intro.items():
            rows.append({
                'scenario': scenario,
                'param_idx': pidx,
                'net_idx': nidx,
                'sim_idx': sidx,
                'state': state,
                'first_intro_day': day,
                'chain_depth': depth,
                'intro_source_state': src_st,
            })

    return pd.DataFrame(rows)


##### 4. Consequential vs dead-end interstate transmissions (summarized) #####

def classify_consequential(trans_df, county_times_df, county_state):
    """Per-sim: fraction of interstate transmissions that were consequential."""
    ct = county_times_df.copy()
    ct['state'] = ct['county'].map(county_state)
    ct = ct.dropna(subset=['state'])

    state_county_counts = ct.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx', 'state']
    )['county'].nunique().reset_index(name='dest_state_total_counties')

    tr = trans_df.copy()
    add_state_columns(tr, county_state)
    interstate = tr[tr['src_state'] != tr['dest_state']].copy()

    if len(interstate) == 0:
        return pd.DataFrame(), pd.DataFrame()

    merged = interstate.merge(
        state_county_counts,
        left_on=['scenario', 'param_idx', 'net_idx', 'sim_idx', 'dest_state'],
        right_on=['scenario', 'param_idx', 'net_idx', 'sim_idx', 'state'],
        how='left',
    )
    merged['dest_state_total_counties'] = merged['dest_state_total_counties'].fillna(1).astype(int)
    merged['is_consequential'] = merged['dest_state_total_counties'] > 1

    per_sim = merged.groupby(
        ['scenario', 'param_idx', 'net_idx', 'sim_idx']
    ).agg(
        n_interstate=('is_consequential', 'count'),
        n_consequential=('is_consequential', 'sum'),
    ).reset_index()
    per_sim['frac_consequential'] = (
        per_sim['n_consequential'] / per_sim['n_interstate'])

    summary = per_sim.groupby('scenario').agg(
        median_n_interstate=('n_interstate', 'median'),
        median_n_consequential=('n_consequential', 'median'),
        median_frac_consequential=('frac_consequential', 'median'),
        q025_frac=('frac_consequential', lambda x: np.percentile(x, 2.5)),
        q975_frac=('frac_consequential', lambda x: np.percentile(x, 97.5)),
        n_sims=('n_interstate', 'count'),
    ).reset_index()

    return per_sim, summary


##### Main #####

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate transmission logs into compact summaries')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--abc_results', type=str, default=None)
    parser.add_argument('--scenarios', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    tx_to_npi_days = config['timeline']['tx_to_npi_days']
    seed_county = config['model']['seed_county']

    output_dir = args.output_dir or os.path.join(args.input_dir, 'aggregates')
    os.makedirs(output_dir, exist_ok=True)

    from sir_model import load_county_data
    Sz, _ = load_county_data(config)
    county_state = build_county_state_map(Sz)

    tmrca_offsets = {}
    abc_path = args.abc_results or os.path.join(args.input_dir, 'abc_accepted.csv')
    if os.path.exists(abc_path):
        abc_df = pd.read_csv(abc_path)
        for idx, row in abc_df.iterrows():
            tmrca_offsets[idx] = int(row['tmrca_offset'])

    batch_dir = os.path.join(args.input_dir, 'batches')
    if not os.path.exists(batch_dir):
        print(f"ERROR: no batches directory at {batch_dir}")
        sys.exit(1)

    batch_files = os.listdir(batch_dir)
    all_scenarios = set()
    for f in batch_files:
        if f.endswith('_transmission_log.csv'):
            name = f.replace('_transmission_log.csv', '')
            parts = name.split('_', 2)
            if len(parts) >= 3:
                all_scenarios.add(parts[2])

    if args.scenarios:
        scenarios = sorted(set(args.scenarios.split(',')) & all_scenarios)
    else:
        scenarios = sorted(all_scenarios)

    batch_indices = sorted({
        int(f.split('_')[1])
        for f in batch_files
        if f.startswith('batch_') and f.endswith('.csv')
        and f.split('_')[1].isdigit()
    })

    print(f"Aggregating {len(scenarios)} scenarios, {len(batch_indices)} batches")
    sys.stdout.flush()

    # Accumulators for per-sim data (medium) and summaries (small)
    acc_sp_persim = []
    acc_sp_summary = []
    acc_wp_summary = []
    acc_cd = []
    acc_cq_persim = []
    acc_cq_summary = []

    for chunk_start in range(0, len(batch_indices), args.batch_size):
        chunk = batch_indices[chunk_start:chunk_start + args.batch_size]
        print(f"\nBatches {chunk[0]}-{chunk[-1]}...", flush=True)

        for scenario in scenarios:
            trans_dfs = []
            county_dfs = []

            for bidx in chunk:
                tag = f'batch_{bidx:03d}_{scenario}'
                tp = os.path.join(batch_dir, f'{tag}_transmission_log.csv')
                cp = os.path.join(batch_dir, f'{tag}_county_times.csv')
                if os.path.exists(tp):
                    trans_dfs.append(pd.read_csv(tp))
                if os.path.exists(cp):
                    county_dfs.append(pd.read_csv(cp))

            if not trans_dfs:
                continue

            trans_df = pd.concat(trans_dfs, ignore_index=True)
            county_df = (pd.concat(county_dfs, ignore_index=True)
                        if county_dfs else None)
            n_trans = len(trans_df)

            # 1. State pairs
            sp_ps, sp_sum = aggregate_state_pairs(
                trans_df, county_state, tmrca_offsets, tx_to_npi_days)
            acc_sp_persim.append(sp_ps)
            acc_sp_summary.append(sp_sum)

            # 2. Weekly profiles (summary only — per-sim is too large)
            _, wp_sum = aggregate_weekly_profiles(trans_df, county_state)
            acc_wp_summary.append(wp_sum)

            # 3. Chain depth
            cd = compute_chain_depth(trans_df, county_state, seed_county)
            acc_cd.append(cd)

            # 4. Consequential
            if county_df is not None and len(county_df) > 0:
                cq_ps, cq_sum = classify_consequential(
                    trans_df, county_df, county_state)
                if len(cq_ps) > 0:
                    acc_cq_persim.append(cq_ps)
                if len(cq_sum) > 0:
                    acc_cq_summary.append(cq_sum)

            print(f"  {scenario}: {n_trans:,} transmissions", flush=True)

            del trans_df, trans_dfs
            if county_df is not None:
                del county_df, county_dfs

    # ── Save ──
    print(f"\nSaving to {output_dir}/...", flush=True)

    def save(name, parts):
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
            print(f"  {name}.csv: {len(df):,} rows")
        else:
            print(f"  {name}.csv: no data")

    save('state_pair_persim', acc_sp_persim)
    save('state_pair_summary', acc_sp_summary)
    save('weekly_profiles_summary', acc_wp_summary)
    save('chain_depth', acc_cd)
    save('consequential_persim', acc_cq_persim)
    save('consequential_summary', acc_cq_summary)

    print("\nAggregation complete.")


if __name__ == '__main__':
    main()
