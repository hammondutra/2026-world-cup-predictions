"""
World Cup 2026 Elo Bracket Simulator + YouTube Visual Asset Generator
Author: ChatGPT technical co-pilot

Inputs:
    optimized_elo_2026.csv with columns: Team, Pre_Tournament_Elo

Outputs:
    /mnt/data/wc26_outputs/
        CSVs with projected standings, qualifiers, bracket, and probabilities
        PNGs for group cards, third-place race, qualification probabilities,
        stage heatmap, champion rankings, official-style bracket, champion path

Important modeling note:
    This script uses the official 2026 tournament structure:
      - 12 groups of 4
      - top 2 from each group qualify
      - 8 best third-place teams qualify
      - Round of 32 official match slots

    FIFA's exact Annex C mapping gives a predetermined placement of third-place
    teams for every possible combination. This script enforces the official
    allowed third-place slots and uses deterministic backtracking to assign
    the qualified third-place groups into those slots. For a polished video MVP,
    this is vastly better than a fake seeded bracket. If you later obtain Annex C
    as a table, replace resolve_third_place_slots() with a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# -----------------------------
# Config
# -----------------------------

BASE_DIR = Path('/mnt/data')
INPUT_CSV = BASE_DIR / 'optimized_elo_2026.csv'
OUTPUT_DIR = BASE_DIR / 'wc26_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_COL = 'Team'
ELO_COL = 'Pre_Tournament_Elo'

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# Official-style Round of 32 slots from FIFA schedule.
# Codes: W = group winner, R = runner-up, T = third-place qualifier.
R32_SLOTS = [
    {'match': 73, 'left': ('R', 'A'), 'right': ('R', 'B')},
    {'match': 74, 'left': ('W', 'E'), 'right': ('T', ['A', 'B', 'C', 'D', 'F'])},
    {'match': 75, 'left': ('W', 'F'), 'right': ('R', 'C')},
    {'match': 76, 'left': ('W', 'C'), 'right': ('R', 'F')},
    {'match': 77, 'left': ('W', 'I'), 'right': ('T', ['C', 'D', 'F', 'G', 'H'])},
    {'match': 78, 'left': ('R', 'E'), 'right': ('R', 'I')},
    {'match': 79, 'left': ('W', 'A'), 'right': ('T', ['C', 'E', 'F', 'H', 'I'])},
    {'match': 80, 'left': ('W', 'L'), 'right': ('T', ['E', 'H', 'I', 'J', 'K'])},
    {'match': 81, 'left': ('W', 'D'), 'right': ('T', ['B', 'E', 'F', 'I', 'J'])},
    {'match': 82, 'left': ('W', 'G'), 'right': ('T', ['A', 'E', 'H', 'I', 'J'])},
    {'match': 83, 'left': ('R', 'K'), 'right': ('R', 'L')},
    {'match': 84, 'left': ('W', 'H'), 'right': ('R', 'J')},
    {'match': 85, 'left': ('W', 'B'), 'right': ('T', ['E', 'F', 'G', 'I', 'J'])},
    {'match': 86, 'left': ('W', 'J'), 'right': ('R', 'H')},
    {'match': 87, 'left': ('W', 'K'), 'right': ('T', ['D', 'E', 'I', 'J', 'L'])},
    {'match': 88, 'left': ('R', 'D'), 'right': ('R', 'G')},
]

R16_PAIRS = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]
QF_PAIRS = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]
SF_PAIRS = [
    (101, 97, 98),
    (102, 99, 100),
]
FINAL_PAIR = (104, 101, 102)

STAGE_ORDER = ['Qualified', 'R16', 'QF', 'SF', 'Final', 'Champion']


# -----------------------------
# Data loading and probabilities
# -----------------------------

def load_ratings(path: Path = INPUT_CSV):
    ratings = pd.read_csv(path)
    required = {TEAM_COL, ELO_COL}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f'Missing required columns in ratings CSV: {missing}')
    ratings = ratings.sort_values(ELO_COL, ascending=False).reset_index(drop=True)
    elo = dict(zip(ratings[TEAM_COL], ratings[ELO_COL]))
    validate_groups(elo)
    return ratings, elo


def validate_groups(elo: dict[str, float]):
    csv_teams = set(elo)
    group_teams = {team for teams in GROUPS.values() for team in teams}
    missing_from_groups = sorted(csv_teams - group_teams)
    missing_from_csv = sorted(group_teams - csv_teams)
    if missing_from_groups or missing_from_csv:
        raise ValueError(
            f'Team mismatch. In CSV not groups: {missing_from_groups}. '
            f'In groups not CSV: {missing_from_csv}.'
        )
    if len(group_teams) != 48:
        raise ValueError(f'Expected 48 unique group teams, got {len(group_teams)}.')


def elo_expected_score(team_a: str, team_b: str, elo: dict[str, float]) -> float:
    diff = elo[team_a] - elo[team_b]
    return 1 / (1 + 10 ** (-diff / 400))


def match_probs(team_a: str, team_b: str, elo: dict[str, float]):
    """Three-way probabilities: A win, draw, B win."""
    p_a_no_draw = elo_expected_score(team_a, team_b, elo)
    diff = abs(elo[team_a] - elo[team_b])

    # Draws are most common when teams are close in strength.
    p_draw = 0.18 + 0.12 * math.exp(-diff / 300)
    p_draw = float(np.clip(p_draw, 0.16, 0.32))

    p_a_win = (1 - p_draw) * p_a_no_draw
    p_b_win = (1 - p_draw) * (1 - p_a_no_draw)
    return p_a_win, p_draw, p_b_win


def simulate_score(outcome: str, rng: np.random.Generator):
    """Simple scoreline generator for standings tiebreakers."""
    if outcome == 'draw':
        goals = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.46, 0.23, 0.06]))
        return goals, goals

    loser_goals = int(rng.choice([0, 1, 2, 3], p=[0.50, 0.33, 0.13, 0.04]))
    margin = int(rng.choice([1, 2, 3, 4], p=[0.57, 0.27, 0.12, 0.04]))
    winner_goals = loser_goals + margin

    if outcome == 'a':
        return winner_goals, loser_goals
    return loser_goals, winner_goals


def simulate_group_match(team_a: str, team_b: str, elo: dict[str, float], rng: np.random.Generator):
    p_a, p_d, p_b = match_probs(team_a, team_b, elo)
    outcome = rng.choice(['a', 'draw', 'b'], p=[p_a, p_d, p_b])
    goals_a, goals_b = simulate_score(outcome, rng)
    return goals_a, goals_b


def knockout_winner(team_a: str, team_b: str, elo: dict[str, float], rng: np.random.Generator):
    p_a = elo_expected_score(team_a, team_b, elo)
    winner = team_a if rng.random() < p_a else team_b
    return winner, p_a


# -----------------------------
# Group stage
# -----------------------------

def simulate_group(group_name: str, teams: list[str], elo: dict[str, float], rng: np.random.Generator):
    rows = {
        team: {
            'Group': group_name,
            'Team': team,
            'Elo': elo[team],
            'MP': 0,
            'W': 0,
            'D': 0,
            'L': 0,
            'GF': 0,
            'GA': 0,
            'GD': 0,
            'Pts': 0,
        }
        for team in teams
    }

    for team_a, team_b in combinations(teams, 2):
        ga, gb = simulate_group_match(team_a, team_b, elo, rng)
        rows[team_a]['MP'] += 1
        rows[team_b]['MP'] += 1
        rows[team_a]['GF'] += ga
        rows[team_a]['GA'] += gb
        rows[team_b]['GF'] += gb
        rows[team_b]['GA'] += ga

        if ga > gb:
            rows[team_a]['W'] += 1
            rows[team_b]['L'] += 1
            rows[team_a]['Pts'] += 3
        elif gb > ga:
            rows[team_b]['W'] += 1
            rows[team_a]['L'] += 1
            rows[team_b]['Pts'] += 3
        else:
            rows[team_a]['D'] += 1
            rows[team_b]['D'] += 1
            rows[team_a]['Pts'] += 1
            rows[team_b]['Pts'] += 1

    df = pd.DataFrame(rows.values())
    df['GD'] = df['GF'] - df['GA']

    # MVP tiebreaker: points, GD, goals scored, Elo.
    # FIFA has further conduct/ranking criteria; Elo is a useful stand-in for final tie breaks.
    df = df.sort_values(['Pts', 'GD', 'GF', 'Elo'], ascending=[False, False, False, False]).reset_index(drop=True)
    df['Pos'] = range(1, 5)
    return df


def simulate_group_stage(elo: dict[str, float], rng: np.random.Generator):
    return pd.concat(
        [simulate_group(g, teams, elo, rng) for g, teams in GROUPS.items()],
        ignore_index=True
    )


def get_qualifiers(standings: pd.DataFrame):
    top_two = standings[standings['Pos'] <= 2].copy()
    thirds = standings[standings['Pos'] == 3].copy()

    thirds_sorted = thirds.sort_values(['Pts', 'GD', 'GF', 'Elo'], ascending=[False, False, False, False]).reset_index(drop=True)
    thirds_sorted['ThirdPlaceRank'] = range(1, len(thirds_sorted) + 1)
    thirds_sorted['ThirdQualified'] = thirds_sorted['ThirdPlaceRank'] <= 8

    best_thirds = thirds_sorted[thirds_sorted['ThirdQualified']].copy()

    qualifiers = pd.concat([top_two, best_thirds], ignore_index=True)
    qualifiers['QualificationType'] = np.where(
        qualifiers['Pos'] == 1, 'Group winner',
        np.where(qualifiers['Pos'] == 2, 'Runner-up', 'Best third-place')
    )
    return qualifiers, thirds_sorted


def group_role_maps(standings: pd.DataFrame, best_thirds: pd.DataFrame):
    roles = {}
    for group, gdf in standings.groupby('Group'):
        sorted_g = gdf.sort_values('Pos')
        roles[('W', group)] = sorted_g.iloc[0]['Team']
        roles[('R', group)] = sorted_g.iloc[1]['Team']
        roles[('T', group)] = sorted_g.iloc[2]['Team']
    qualified_third_groups = set(best_thirds[best_thirds['ThirdQualified']]['Group'])
    return roles, qualified_third_groups


# -----------------------------
# Official-style bracket construction
# -----------------------------

def resolve_third_place_slots(qualified_third_groups: set[str]):
    """
    Assign qualified third-place groups to the official third-place slots.

    This respects the allowed group sets in each FIFA match slot. FIFA's Annex C
    specifies one official mapping for every combination; this backtracking solver
    finds a valid mapping for the same slot constraints.
    """
    slots = []
    for slot in R32_SLOTS:
        for side in ['left', 'right']:
            code, value = slot[side]
            if code == 'T':
                allowed = [g for g in value if g in qualified_third_groups]
                slots.append({'match': slot['match'], 'side': side, 'allowed': allowed})

    if len(slots) != 8 or len(qualified_third_groups) != 8:
        raise ValueError('Expected exactly 8 third-place slots and 8 qualified third-place groups.')

    # Backtracking, trying constrained slots first.
    ordered = sorted(slots, key=lambda s: (len(s['allowed']), s['match']))
    assignment = {}

    def backtrack(i, used):
        if i == len(ordered):
            return True
        slot = ordered[i]
        # Use official listed order where possible, but only remaining groups.
        for group in slot['allowed']:
            if group in used:
                continue
            assignment[(slot['match'], slot['side'])] = group
            used.add(group)
            if backtrack(i + 1, used):
                return True
            used.remove(group)
            assignment.pop((slot['match'], slot['side']), None)
        return False

    if not backtrack(0, set()):
        raise RuntimeError(f'No valid third-place slot assignment found for {sorted(qualified_third_groups)}')
    return assignment


def resolve_slot(slot_side, roles, third_assignment):
    code, value = slot_side
    if code in ['W', 'R']:
        group = value
        return roles[(code, group)], f'{code}{group}'

    if code == 'T':
        raise ValueError('Third-place slots must be resolved with match+side context.')

    raise ValueError(f'Unknown slot code: {code}')


def make_round_of_32(standings: pd.DataFrame, thirds_sorted: pd.DataFrame):
    roles, qualified_third_groups = group_role_maps(standings, thirds_sorted)
    third_assignment = resolve_third_place_slots(qualified_third_groups)

    matches = []
    for slot in R32_SLOTS:
        match_no = slot['match']
        sides = []
        labels = []
        for side_name in ['left', 'right']:
            code, value = slot[side_name]
            if code == 'T':
                group = third_assignment[(match_no, side_name)]
                team = roles[('T', group)]
                label = f'3{group}'
            else:
                group = value
                team = roles[(code, group)]
                label = f'{code}{group}'
            sides.append(team)
            labels.append(label)

        matches.append({
            'Match': match_no,
            'Round': 'Round of 32',
            'Slot A': labels[0],
            'Team A': sides[0],
            'Slot B': labels[1],
            'Team B': sides[1],
        })
    return pd.DataFrame(matches)


def play_match_row(match_no, round_name, team_a, team_b, elo, rng):
    winner, p_a = knockout_winner(team_a, team_b, elo, rng)
    return {
        'Match': match_no,
        'Round': round_name,
        'Team A': team_a,
        'Team B': team_b,
        'P(Team A advances)': p_a,
        'P(Team B advances)': 1 - p_a,
        'Winner': winner,
    }


def play_bracket(r32_df: pd.DataFrame, elo: dict[str, float], rng: np.random.Generator):
    rows = []
    winners = {}

    for _, row in r32_df.iterrows():
        result = play_match_row(row['Match'], 'Round of 32', row['Team A'], row['Team B'], elo, rng)
        result['Slot A'] = row['Slot A']
        result['Slot B'] = row['Slot B']
        rows.append(result)
        winners[row['Match']] = result['Winner']

    round_specs = [
        ('Round of 16', R16_PAIRS),
        ('Quarterfinals', QF_PAIRS),
        ('Semifinals', SF_PAIRS),
        ('Final', [FINAL_PAIR]),
    ]

    for round_name, pairs in round_specs:
        for match_no, left_match, right_match in pairs:
            team_a = winners[left_match]
            team_b = winners[right_match]
            result = play_match_row(match_no, round_name, team_a, team_b, elo, rng)
            result['Source A'] = f'W{left_match}'
            result['Source B'] = f'W{right_match}'
            rows.append(result)
            winners[match_no] = result['Winner']

    bracket = pd.DataFrame(rows)
    champion = winners[104]
    return champion, bracket


def simulate_tournament(elo, seed=7):
    rng = np.random.default_rng(seed)
    standings = simulate_group_stage(elo, rng)
    qualifiers, thirds_sorted = get_qualifiers(standings)
    r32 = make_round_of_32(standings, thirds_sorted)
    champion, bracket = play_bracket(r32, elo, rng)
    return standings, qualifiers, thirds_sorted, r32, bracket, champion


# -----------------------------
# Monte Carlo
# -----------------------------

def run_monte_carlo(elo, n_sims=3000, seed=42):
    rng = np.random.default_rng(seed)

    teams = list(elo.keys())
    stage_counts = {team: Counter() for team in teams}
    points_sum = Counter()
    gd_sum = Counter()
    finish_pos_counts = {team: Counter() for team in teams}

    for _ in range(n_sims):
        standings = simulate_group_stage(elo, rng)
        qualifiers, thirds_sorted = get_qualifiers(standings)
        r32 = make_round_of_32(standings, thirds_sorted)
        champion, bracket = play_bracket(r32, elo, rng)

        for _, row in standings.iterrows():
            team = row['Team']
            points_sum[team] += row['Pts']
            gd_sum[team] += row['GD']
            finish_pos_counts[team][int(row['Pos'])] += 1

        qualified = set(qualifiers['Team'])
        r16 = set(bracket[bracket['Round'] == 'Round of 32']['Winner'])
        qf = set(bracket[bracket['Round'] == 'Round of 16']['Winner'])
        sf = set(bracket[bracket['Round'] == 'Quarterfinals']['Winner'])
        finalists = set(bracket[bracket['Round'] == 'Semifinals']['Winner'])

        for team in qualified:
            stage_counts[team]['Qualified'] += 1
        for team in r16:
            stage_counts[team]['R16'] += 1
        for team in qf:
            stage_counts[team]['QF'] += 1
        for team in sf:
            stage_counts[team]['SF'] += 1
        for team in finalists:
            stage_counts[team]['Final'] += 1
        stage_counts[champion]['Champion'] += 1

    stage_rows = []
    for team in teams:
        row = {
            'Team': team,
            'Elo': elo[team],
            'Avg Points': points_sum[team] / n_sims,
            'Avg GD': gd_sum[team] / n_sims,
            'Group 1st %': finish_pos_counts[team][1] / n_sims,
            'Group 2nd %': finish_pos_counts[team][2] / n_sims,
            'Group 3rd %': finish_pos_counts[team][3] / n_sims,
            'Group 4th %': finish_pos_counts[team][4] / n_sims,
        }
        for stage in STAGE_ORDER:
            row[f'{stage} %'] = stage_counts[team][stage] / n_sims
        stage_rows.append(row)

    stage_probs = pd.DataFrame(stage_rows).sort_values('Champion %', ascending=False).reset_index(drop=True)
    champion_probs = stage_probs[['Team', 'Elo', 'Champion %', 'Final %', 'SF %', 'QF %', 'R16 %', 'Qualified %']].copy()
    return stage_probs, champion_probs


# -----------------------------
# Visualization utilities
# -----------------------------

def save_group_cards(standings, output_path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()

    for ax, group in zip(axes, sorted(GROUPS.keys())):
        ax.axis('off')
        gdf = standings[standings['Group'] == group].sort_values('Pos')
        display_df = gdf[['Pos', 'Team', 'Pts', 'GD', 'GF']].copy()
        display_df['Status'] = display_df['Pos'].map({1: 'Q', 2: 'Q', 3: '3rd race', 4: 'Out'})
        display_df = display_df[['Pos', 'Team', 'Pts', 'GD', 'GF', 'Status']]

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#222222')
                cell.get_text().set_color('white')
            elif row in [1, 2]:
                cell.set_facecolor('#D9F2D9')
            elif row == 3:
                cell.set_facecolor('#FFF0CC')
            else:
                cell.set_facecolor('#F5F5F5')

        ax.set_title(f'Group {group}', fontsize=14, weight='bold', pad=8)

    fig.suptitle('Projected Group Stage Standings — Single Bracket Example', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_third_place_race(thirds_sorted, output_path):
    df = thirds_sorted.copy().sort_values('ThirdPlaceRank', ascending=True)
    labels = [f"{r.Team} ({r.Group})" for _, r in df.iterrows()]
    values = df['Pts'].values
    colors = ['#2ca02c' if q else '#d62728' for q in df['ThirdQualified']]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(df.iloc[7]['Pts'], linestyle='--', linewidth=1.5, color='#555555')
    ax.set_xlabel('Points')
    ax.set_title('Best Third-Place Race — Top 8 Advance', fontsize=18, weight='bold')
    ax.text(0.98, 0.04, 'Green = qualifies · Red = eliminated', transform=ax.transAxes, ha='right', fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_points_gd(stage_probs, output_path):
    df = stage_probs.sort_values('Avg Points', ascending=False).head(24).copy()
    fig, ax = plt.subplots(figsize=(13, 8))
    sizes = 60 + df['Qualified %'] * 500
    scatter = ax.scatter(df['Avg Points'], df['Avg GD'], s=sizes, alpha=0.72)
    for _, r in df.iterrows():
        ax.text(r['Avg Points'] + 0.015, r['Avg GD'] + 0.015, r['Team'], fontsize=8)
    ax.set_xlabel('Average group-stage points')
    ax.set_ylabel('Average goal difference')
    ax.set_title('Who Looks Strongest in the Group Stage?', fontsize=18, weight='bold')
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.03, 'Bubble size = qualification probability', transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_qualification_probability(stage_probs, output_path):
    df = stage_probs.sort_values('Qualified %', ascending=True)
    labels = df['Team']
    values = df['Qualified %'] * 100
    colors = ['#2ca02c' if v >= 75 else '#ffbf00' if v >= 45 else '#d62728' for v in values]

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Qualification probability (%)')
    ax.set_title('Chance to Reach the Round of 32', fontsize=18, weight='bold')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_stage_heatmap(stage_probs, output_path, top_n=20):
    df = stage_probs.sort_values('Champion %', ascending=False).head(top_n).copy()
    stages = ['Qualified %', 'R16 %', 'QF %', 'SF %', 'Final %', 'Champion %']
    values = df[stages].values * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(values, aspect='auto', cmap='YlGnBu')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Team'])
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(['R32', 'R16', 'QF', 'SF', 'Final', 'Win'])
    ax.set_title('Tournament Stage Probability Heatmap', fontsize=18, weight='bold')

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f'{values[i, j]:.0f}%', ha='center', va='center', fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability (%)')
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_champion_ranking(champion_probs, output_path, top_n=16):
    df = champion_probs.sort_values('Champion %', ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(df['Team'], df['Champion %'] * 100, color='#1f77b4')
    ax.set_xlabel('Champion probability (%)')
    ax.set_title('Most Likely World Cup Champions', fontsize=18, weight='bold')
    ax.grid(axis='x', alpha=0.25)
    for i, v in enumerate(df['Champion %'] * 100):
        ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _wrap_team(name, max_width=16):
    return '\n'.join(textwrap.wrap(name, max_width))


def save_bracket_graphic(bracket, output_path):
    # Custom bracket layout by official match paths.
    # y positions for R32 are 15..0. Later rounds inherit mean y of source matches.
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 16.5)

    x_by_round = {
        'Round of 32': 0.4,
        'Round of 16': 3.2,
        'Quarterfinals': 6.0,
        'Semifinals': 8.6,
        'Final': 10.7,
    }
    box_w = 2.15
    box_h = 0.62

    match_y = {}
    r32_matches = [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]
    # Arrange by the actual next-round tree instead of chronological order.
    tree_order = [74,77,73,75,76,78,79,80,83,84,81,82,86,88,85,87]
    for idx, m in enumerate(tree_order):
        match_y[m] = 15 - idx

    def draw_box(x, y, match, round_name, team_a, team_b, winner):
        rect = Rectangle((x, y - box_h / 2), box_w, box_h, facecolor='#f7f7f7', edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        team_a_text = _wrap_team(team_a)
        team_b_text = _wrap_team(team_b)
        text = f'M{match}\n{team_a_text} vs {team_b_text}\n→ {_wrap_team(winner)}'
        ax.text(x + box_w / 2, y, text, ha='center', va='center', fontsize=7.4)

    rows_by_match = {int(r['Match']): r for _, r in bracket.iterrows()}

    # Draw R32 boxes.
    for m in tree_order:
        r = rows_by_match[m]
        draw_box(x_by_round['Round of 32'], match_y[m], m, 'Round of 32', r['Team A'], r['Team B'], r['Winner'])

    # Draw later boxes and connectors.
    pair_sets = [('Round of 16', R16_PAIRS), ('Quarterfinals', QF_PAIRS), ('Semifinals', SF_PAIRS), ('Final', [FINAL_PAIR])]
    for round_name, pairs in pair_sets:
        x = x_by_round[round_name]
        for m, src1, src2 in pairs:
            y = (match_y[src1] + match_y[src2]) / 2
            match_y[m] = y
            r = rows_by_match[m]
            draw_box(x, y, m, round_name, r['Team A'], r['Team B'], r['Winner'])

            # connectors from source boxes to current box
            src_round = 'Round of 32'
            x_prev_candidates = [xr for rn, xr in x_by_round.items() if xr < x]
            x_prev = max(x_prev_candidates)
            for src in [src1, src2]:
                ax.plot([x_prev + box_w, x - 0.08], [match_y[src], y], color='#555555', linewidth=0.8, alpha=0.7)

    ax.text(0.4, 16.25, 'Official-Style 2026 World Cup Knockout Bracket Example', fontsize=24, weight='bold')
    ax.text(0.4, 15.85, 'Round of 32 slots follow the 2026 format; arrows show simulated winners from one example run.', fontsize=12)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_champion_path(bracket, champion, output_path):
    path = bracket[bracket['Winner'] == champion].copy()
    path = path.sort_values('Match')
    # Better chronological by round order.
    order_map = {'Round of 32': 1, 'Round of 16': 2, 'Quarterfinals': 3, 'Semifinals': 4, 'Final': 5}
    path['RoundOrder'] = path['Round'].map(order_map)
    path = path.sort_values('RoundOrder')

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')
    ax.set_xlim(0, len(path) * 2.2)
    ax.set_ylim(0, 2)

    for i, (_, r) in enumerate(path.iterrows()):
        x = i * 2.2 + 0.2
        opponent = r['Team B'] if r['Team A'] == champion else r['Team A']
        p = r['P(Team A advances)'] if r['Team A'] == champion else r['P(Team B advances)']
        rect = Rectangle((x, 0.55), 1.75, 0.8, facecolor='#f7f7f7', edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.875, 0.95, f"{r['Round']}\nvs {_wrap_team(opponent, 14)}\nWin chance: {p*100:.0f}%", ha='center', va='center', fontsize=9)
        if i < len(path) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.78, 0.95), (x + 2.15, 0.95), arrowstyle='->', mutation_scale=12, linewidth=1.2))

    ax.text(0.2, 1.72, f'{champion}: Path to the Trophy', fontsize=22, weight='bold')
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Main generation
# -----------------------------

def main():
    ratings, elo = load_ratings()

    # One fixed example for the video bracket.
    standings, qualifiers, thirds_sorted, r32, bracket, champion = simulate_tournament(elo, seed=14)

    # Monte Carlo probabilities for credibility overlays.
    stage_probs, champion_probs = run_monte_carlo(elo, n_sims=600, seed=2026)

    # Save tables.
    ratings.to_csv(OUTPUT_DIR / 'input_elo_ratings_sorted.csv', index=False)
    standings.to_csv(OUTPUT_DIR / 'example_group_standings.csv', index=False)
    thirds_sorted.to_csv(OUTPUT_DIR / 'example_third_place_ranking.csv', index=False)
    qualifiers.to_csv(OUTPUT_DIR / 'example_qualified_teams.csv', index=False)
    r32.to_csv(OUTPUT_DIR / 'example_round_of_32_slots.csv', index=False)
    bracket.to_csv(OUTPUT_DIR / 'example_full_knockout_bracket.csv', index=False)
    stage_probs.to_csv(OUTPUT_DIR / 'monte_carlo_stage_probabilities.csv', index=False)
    champion_probs.to_csv(OUTPUT_DIR / 'monte_carlo_champion_probabilities.csv', index=False)

    # Save visuals.
    save_group_cards(standings, OUTPUT_DIR / '01_group_stage_standings_grid.png')
    save_third_place_race(thirds_sorted, OUTPUT_DIR / '02_best_third_place_race.png')
    save_points_gd(stage_probs, OUTPUT_DIR / '03_expected_points_vs_goal_difference.png')
    save_qualification_probability(stage_probs, OUTPUT_DIR / '04_qualification_probability_all_teams.png')
    save_stage_heatmap(stage_probs, OUTPUT_DIR / '05_stage_probability_heatmap_top20.png')
    save_champion_ranking(champion_probs, OUTPUT_DIR / '06_champion_probability_ranking.png')
    save_bracket_graphic(bracket, OUTPUT_DIR / '07_official_style_knockout_bracket.png')
    save_champion_path(bracket, champion, OUTPUT_DIR / '08_champion_path_to_final.png')

    print(f'Generated outputs in: {OUTPUT_DIR}')
    print(f'Example bracket champion: {champion}')
    print('\nTop 10 champion probabilities:')
    print(champion_probs.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
