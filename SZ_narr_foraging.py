

from __future__ import annotations

import ast
import pickle
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPAN = 20
ELLIPSES_BREAK = True
N_TOPICS = [125, 100, 75, 50]
WINDOW_SIZES = [5,7,3]

WORKING_DIR = Path("/")
FORAGING_SAVE_DIR = WORKING_DIR / "SZ_foraging_atoms"
EMBEDDING_DIR = Path("")
DATA_DIR = Path("")
MODEL_DIR = Path(
    f"/MODEL_DIR/"
    f"{SPAN}_{int(SPAN * 0.2)}_SZ_MODEL"
)

FORAGING_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_windowed_similarity_trajectory(
	embeddings: Sequence[np.ndarray],
	window_size: int = 5,
) -> List[float]:
	"""Sliding-window similarity between successive centroids."""

	sims: List[float] = []

	if not embeddings:
		return sims

	for i in range(1, len(embeddings)):
		past_start = max(0, i - window_size)
		past_window = np.asarray(embeddings[past_start:i], dtype=float)
		next_window = np.asarray(embeddings[i : i + window_size], dtype=float)

		if past_window.size == 0 or next_window.size == 0:
			sims.append(np.nan)
			continue

		past_centroid = past_window.mean(axis=0, keepdims=True)
		next_centroid = next_window.mean(axis=0, keepdims=True)
		sims.append(float(cosine_similarity(next_centroid, past_centroid)[0, 0]))

	return sims


def zscore_list(values: Iterable[float]) -> List[float]:
	"""Z-score ``values`` while keeping NaNs at zero."""

	arr = np.asarray(list(values), dtype=float)
	if arr.size == 0:
		return []

	valid = ~np.isnan(arr)
	if valid.sum() < 2:
		return [0.0] * arr.size

	mean = arr[valid].mean()
	std = arr[valid].std()
	if std == 0 or np.isnan(std):
		return [0.0] * arr.size

	z = np.zeros_like(arr)
	z[valid] = (arr[valid] - mean) / std
	return z.tolist()


def delta_similarity_segmentation(
	z_sims: Sequence[float],
	rise_threshold: float,
	fall_threshold: float,
) -> List[str]:
	"""Segment z-scored similarities into ``cluster`` vs ``switch`` states."""

	if not z_sims:
		return []

	sanitized = [0.0 if np.isnan(val) else float(val) for val in z_sims]
	median_sim = float(np.median(sanitized))

	states: List[str] = []
	last_state = "cluster" if sanitized[0] >= median_sim else "switch"
	states.append(last_state)

	for prev, curr in zip(sanitized, sanitized[1:]):
		if last_state == "cluster":
			state = "switch" if prev - curr > fall_threshold else "cluster"
		else:
			state = "cluster" if curr - prev > rise_threshold else "switch"
		states.append(state)
		last_state = state

	return states


def get_cluster_stats(noun_states: Sequence[str], atoms: Sequence[int]) -> Tuple[List[int], float, float, List[float]]:
	n_len = 1
	cluster_lengths: List[int] = []
	atoms_visited: List[float] = []
	atoms_list: List[int] = []
	prev_state = "cluster"

	for n in range(len(noun_states) - 1):
		a = atoms[n]
		state = noun_states[n]
		next_state = noun_states[n + 1]

		if state == "cluster":
			n_len += 1
			atoms_list.append(a)
			prev_state = "cluster"

		elif prev_state == "cluster" and state == "switch": #Case where it is last in cluster
			atoms_list.append(a)
			cluster_lengths.append(n_len)
			atoms_visited.append(len(set(atoms_list)) / n_len if n_len else np.nan)
			n_len = 1
			atoms_list = [a]
			prev_state = "switch"

		elif next_state == "cluster" and state == "switch": #Case where it is first in cluster
			n_len = 1
			atoms_list = [a]
			prev_state = "cluster"

		else:
			continue

	final_a = atoms[-1]
	final_state = noun_states[-1]
	if final_state == "cluster":
		n_len += 1
		atoms_list.append(final_a)
		cluster_lengths.append(n_len)
		atoms_visited.append(len(set(atoms_list)) / n_len if n_len else np.nan)
	elif prev_state == "cluster":
		cluster_lengths.append(n_len)
		atoms_visited.append(len(set(atoms_list)) / n_len if n_len else np.nan)

	mean_length = float(np.mean(cluster_lengths)) if cluster_lengths else np.nan
	median_length = float(np.median(cluster_lengths)) if cluster_lengths else np.nan
	return cluster_lengths, mean_length, median_length, atoms_visited


def get_cluster_cosine_stats(
	noun_states: Sequence[str],
	noun_embs: Sequence[np.ndarray],
) -> Tuple[List[float], List[float], List[np.ndarray], List[List[np.ndarray]]]:
	within_clust_jumps: List[float] = []
	switching_jumps: List[float] = []
	list_clusts: List[List[np.ndarray]] = []
	list_clust_embs: List[np.ndarray] = []

	embs_in_clust: List[np.ndarray] = []
	prev_state: str | None = None

	for n in range(len(noun_states)):
		state = noun_states[n]
		emb = noun_embs[n]

		if state == "cluster":
			embs_in_clust.append(emb)
			within_clust_jumps.append(
				float(
					cosine_similarity(emb.reshape(1, -1), noun_embs[n - 1].reshape(1, -1))[0, 0]
				)
			)
			prev_state = "cluster"
		elif state == "switch":
			switching_jumps.append(
				float(
					cosine_similarity(emb.reshape(1, -1), noun_embs[n - 1].reshape(1, -1))[0, 0]
				)
			)
			if prev_state == "cluster" and embs_in_clust:
				list_clusts.append(embs_in_clust)
				list_clust_embs.append(np.mean(embs_in_clust, axis=0))
				embs_in_clust = []
			prev_state = "switch"

	if prev_state == "cluster" and embs_in_clust:
		list_clusts.append(embs_in_clust)
		list_clust_embs.append(np.mean(embs_in_clust, axis=0))

	return within_clust_jumps, switching_jumps, list_clust_embs, list_clusts


def seq_features(seq: Sequence[int]) -> pd.Series:
	metrics = [
		"entropy",
		"repeat_ratio_contig",
	]

	if not seq or len(seq) < 2:
		return pd.Series([np.nan] * len(metrics), index=metrics)

	seq_list = list(seq)
	n = len(seq_list)

	counts = Counter(seq_list)
	probs = np.array(list(counts.values()), dtype=float) / n
	with np.errstate(divide="ignore", invalid="ignore"):
		H_bits = float(-(probs * np.log2(probs)).sum())
	max_ent_bits = log2(len(counts)) if counts else 0.0
	entropy = H_bits / max_ent_bits if max_ent_bits > 0 else np.nan

	unique_contig = [seq_list[0]]
	for item in seq_list[1:]:
		if item != unique_contig[-1]:
			unique_contig.append(item)
	counts_contig = Counter(unique_contig)
	num_repeats_contig = len(unique_contig) - len(counts_contig)
	repeat_ratio_contig = num_repeats_contig / len(unique_contig)

	return pd.Series([entropy, repeat_ratio_contig], index=metrics)


def analyze_repetitions_per_row(utterances: Sequence[str]) -> pd.Series:
	if not utterances:
		return pd.Series(
			{
				"prop_repeated_utterances": np.nan,
				"prop_repeated_bigrams": np.nan,
				"prop_repeated_trigrams": np.nan,
				"total_utterances": 0,
				"unique_utterances": 0,
				"total_words": 0,
			}
		)

	total_utterances = len(utterances)
	unique_utterances = len(set(utterances))
	prop_repeated_utterances = (
		(total_utterances - unique_utterances) / total_utterances
		if total_utterances
		else np.nan
	)

	all_words: List[str] = []
	for utt in utterances:
		all_words.extend(utt.split())

	def repetition_ratio(tokens: Sequence[Tuple[str, ...]]) -> float:
		if not tokens:
			return np.nan
		total = len(tokens)
		unique = len(set(tokens))
		return (total - unique) / total if total else np.nan

	bigrams = [tuple(all_words[i : i + 2]) for i in range(len(all_words) - 1)]
	trigrams = [tuple(all_words[i : i + 3]) for i in range(len(all_words) - 2)]

	return pd.Series(
		{
			"prop_repeated_utterances": prop_repeated_utterances,
			"prop_repeated_bigrams": repetition_ratio(bigrams),
			"prop_repeated_trigrams": repetition_ratio(trigrams),
			"total_utterances": total_utterances,
			"unique_utterances": unique_utterances,
			"total_words": len(all_words),
		}
	)


def cardinality(seq: Sequence[int]) -> float:
	seq = list(seq)
	return len(set(seq)) / len(seq) if seq else float("nan")


def smooth_embeddings(traj: Sequence[np.ndarray], window: int = 3) -> np.ndarray:
	"""Causal moving-average smoothing with consistent length."""

	arr = np.asarray(traj, dtype=float)
	if arr.ndim != 2 or window <= 1 or arr.shape[0] < window:
		return arr

	smoothed = []
	for idx in range(arr.shape[0]):
		start = max(0, idx - window + 1)
		smoothed.append(arr[start : idx + 1].mean(axis=0))
	return np.vstack(smoothed)


def get_age_loading(
	df_age: pd.DataFrame,
	age_dimension: np.ndarray,
	smooth_traj: bool = True,
) -> pd.DataFrame:
	mean_diffs, std_diffs, cv_diffs = [], [], []
	net_changes, total_paths, progress = [], [], []
	monotonicities, curvatures = [], []

	for traj in df_age["embeddings"]:
		traj_arr = np.asarray(traj, dtype=float)
		if traj_arr.ndim != 2 or traj_arr.size == 0:
			mean_diffs.append(np.nan)
			std_diffs.append(np.nan)
			cv_diffs.append(np.nan)
			net_changes.append(np.nan)
			total_paths.append(np.nan)
			progress.append(np.nan)
			monotonicities.append(np.nan)
			curvatures.append(np.nan)
			continue

		if smooth_traj:
			traj_arr = smooth_embeddings(traj_arr)

		loadings = cosine_similarity(traj_arr, age_dimension.reshape(1, -1)).ravel()

		if loadings.size > 1:
			diffs = np.abs(np.diff(loadings))
			mean_step = float(np.mean(diffs))
			std_step = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
			cv_step = std_step / mean_step if mean_step else 0.0

			net_change = float(loadings[-1] - loadings[0])
			total_path = float(np.sum(diffs))
			progress_ratio = net_change / total_path if total_path else 0.0
			monotonicity = float(np.mean(np.diff(loadings) >= 0))
			curvature = (
				float(np.mean(np.abs(np.diff(loadings, n=2))))
				if loadings.size > 2
				else 0.0
			)
		else:
			mean_step = std_step = cv_step = 0.0
			net_change = total_path = progress_ratio = 0.0
			monotonicity = curvature = 0.0

		mean_diffs.append(mean_step)
		std_diffs.append(std_step)
		cv_diffs.append(cv_step)
		net_changes.append(net_change)
		total_paths.append(total_path)
		progress.append(progress_ratio)
		monotonicities.append(monotonicity)
		curvatures.append(curvature)

	df_age = df_age.copy()
	df_age["age_step_mean"] = mean_diffs
	df_age["age_step_sd"] = std_diffs
	df_age["age_step_cv"] = cv_diffs
	df_age["age_net_change"] = net_changes
	df_age["age_total_path"] = total_paths
	df_age["age_progress_ratio"] = progress
	df_age["age_monotonicity"] = monotonicities
	df_age["age_curvature"] = curvatures
	return df_age


# ---------------------------------------------------------------------------
# Segmentation core
# ---------------------------------------------------------------------------


def main_seg(
	df: pd.DataFrame,
	data_column_name: str,
	rise_threshold: float,
	fall_threshold: float,
	window_size: int,
	verbose: bool,
) -> pd.DataFrame:
	noun_sim_states = []
	all_cluster_lengths = []
	all_mean_length = []
	mean_atoms_visited = []
	all_within_clust_jumps = []
	all_switching_jumps = []

	for row_idx, embeddings in tqdm(
		enumerate(df[data_column_name]), total=len(df[data_column_name])
	):
		atom_seq = df.loc[row_idx, "atom_seq"]
		utterances = df.loc[row_idx, "utterances"] if "utterances" in df.columns else None

		if embeddings is None:
			noun_sim_states.append(np.nan)
			all_cluster_lengths.append(np.nan)
			all_mean_length.append(np.nan)
			mean_atoms_visited.append(np.nan)
			all_within_clust_jumps.append(np.nan)
			all_switching_jumps.append(np.nan)
			continue

		if isinstance(embeddings, np.ndarray):
			emb_iterable = embeddings
		elif isinstance(embeddings, (list, tuple)):
			emb_iterable = embeddings
		else:
			noun_sim_states.append(np.nan)
			all_cluster_lengths.append(np.nan)
			all_mean_length.append(np.nan)
			mean_atoms_visited.append(np.nan)
			all_within_clust_jumps.append(np.nan)
			all_switching_jumps.append(np.nan)
			continue

		if len(emb_iterable) < 10:
			noun_sim_states.append(np.nan)
			all_cluster_lengths.append(np.nan)
			all_mean_length.append(np.nan)
			mean_atoms_visited.append(np.nan)
			all_within_clust_jumps.append(np.nan)
			all_switching_jumps.append(np.nan)
			continue

		emb_list = [np.asarray(vec, dtype=float) for vec in emb_iterable]

		windowed_sims = calc_windowed_similarity_trajectory(emb_list, window_size)

		windowed_array = np.asarray(windowed_sims, dtype=float)
		fill_value = float(np.nanmean(windowed_array))
		smoothed_input = np.nan_to_num(windowed_array, nan=fill_value)
		smoothed_sims = gaussian_filter1d(smoothed_input, sigma=0.5, mode="nearest")
		z_input = smoothed_sims.tolist()

		z_scores = zscore_list(z_input)
		states = delta_similarity_segmentation(z_scores, rise_threshold, fall_threshold)
		noun_sim_states.append(states)

		if verbose and states and utterances is not None:
			for state, utt in zip(states, utterances):
				print(state, utt)

		cluster_lengths, mean_length, _, atoms_per_cluster = get_cluster_stats(states, atom_seq)
		within_jumps, switching_jumps, _, _ = get_cluster_cosine_stats(states, emb_list)

		all_cluster_lengths.append(cluster_lengths if cluster_lengths else np.nan)
		all_mean_length.append(mean_length)
		mean_atoms_visited.append(float(np.nanmean(atoms_per_cluster)) if atoms_per_cluster else np.nan)

		all_within_clust_jumps.append(float(np.nanmedian(within_jumps)) if within_jumps else np.nan)
		all_switching_jumps.append(float(np.nanmedian(switching_jumps)) if switching_jumps else np.nan)

	df = df.copy()
	df["noun_sim_states"] = noun_sim_states
	df["all_cluster_lengths"] = all_cluster_lengths
	df["mean_cluster_length"] = all_mean_length
	df["mean_atoms_visited"] = mean_atoms_visited
	df["all_within_clust_jumps"] = all_within_clust_jumps
	df["all_switching_jumps"] = all_switching_jumps
	return df


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


@dataclass
class ProcessingConfig:
	n_topics: int
	window_size: int

	@property
	def prefix(self) -> str:
		return "" if ELLIPSES_BREAK else "nellip_"

	@property
	def preprefix(self) -> str:
		return "sm_"


def load_embeddings(prefix: str) -> List[Sequence[np.ndarray]]:
	embedding_path = EMBEDDING_DIR / f"{prefix}IPII_utt_embeddings_{SPAN}.pkl"
	with embedding_path.open("rb") as f:
		return pickle.load(f)


def build_age_dimension() -> np.ndarray:
	sentence_model = SentenceTransformer(str(MODEL_DIR))
	childhood_terms = ["young", "start", "past"]
	old_age_terms = ["old", "end", "present"]
	child_vecs = np.mean([sentence_model.encode(term) for term in childhood_terms], axis=0)
	old_vecs = np.mean([sentence_model.encode(term) for term in old_age_terms], axis=0)
	return old_vecs - child_vecs


def process_config(cfg: ProcessingConfig, age_dimension: np.ndarray, verbose: bool = True, test_mode: bool = False) -> None:
	csv_path = DATA_DIR / f"{cfg.prefix}IPIIs_censored_utterances_{cfg.n_topics}.csv"
	df_texts = pd.read_csv(csv_path)
	df_texts["utterances"] = df_texts["utterances"].apply(ast.literal_eval)
	repetition_stats = df_texts["utterances"].apply(analyze_repetitions_per_row)
	df_texts = pd.concat([df_texts, repetition_stats], axis=1)
	print(len(df_texts), "utterances loaded")

	df_texts["atom_seq"] = df_texts["atom_seq"].apply(ast.literal_eval)
	df_texts[[
		"entropy",
		"repeat_ratio_contig",
	]] = df_texts["atom_seq"].apply(seq_features)

	embeddings = load_embeddings(cfg.prefix)
	if len(embeddings) != len(df_texts):
		raise ValueError("Embedding list length does not match dataframe rows")

	df_texts["embeddings"] = embeddings
	df_texts["num_atoms"] = df_texts["atom_seq"].apply(cardinality)
	df_texts = get_age_loading(df_texts, age_dimension)
	df_texts["length"] = df_texts["embeddings"].apply(len)
	if test_mode:
		df_texts = df_texts[0:3]
	df_texts = main_seg(
		df_texts,
		data_column_name="embeddings",
		rise_threshold=0.0,
		fall_threshold=1.0,
		window_size=cfg.window_size,
		verbose=verbose,
	)
	print(len(df_texts), "utterances loaded")

	df_texts = df_texts.drop(columns=["embeddings"], errors="ignore")
	df_texts = df_texts.drop(columns=["sentence_embeddings"], errors="ignore")

	print(df_texts)
	if not test_mode:
		output_name = f"{cfg.preprefix}{cfg.prefix}utt_seg_means_{SPAN}_{cfg.window_size}_{cfg.n_topics}_clean.csv"
		df_texts.to_csv(FORAGING_SAVE_DIR / output_name, index=False)


def main(verbose: bool = True) -> None:
	age_dimension = build_age_dimension()

	for n_topics in N_TOPICS:
		for window_size in WINDOW_SIZES:
			cfg = ProcessingConfig(
				n_topics=n_topics,
				window_size=window_size,
			)
			process_config(cfg, age_dimension, verbose=verbose, test_mode=False)


if __name__ == "__main__":
	print("Ensure that the postdoc environment is activated")
	main(verbose=False)
