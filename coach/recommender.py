from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class Recommendation:
    exercise_name: str
    muscle_group: str
    difficulty: str
    reason: str


class ExerciseRecommender:
    def __init__(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        self.nn = NearestNeighbors(n_neighbors=8, metric="cosine")

        corpus = (self.df["goal_tags"].fillna("") + " " + self.df["muscle_group"].fillna("")).tolist()
        X = self.vectorizer.fit_transform(corpus)
        self.nn.fit(X)

    def recommend(self, goals: str, focus: str = "", difficulty: str = "", top_k: int = 6) -> List[Recommendation]:
        query = f"{goals} {focus}".strip()
        qv = self.vectorizer.transform([query])
        distances, indices = self.nn.kneighbors(qv, n_neighbors=min(top_k, len(self.df)))

        recs: List[Recommendation] = []
        for d, idx in zip(distances[0], indices[0]):
            row = self.df.iloc[int(idx)]
            if difficulty and str(row["difficulty"]).lower() != difficulty.lower():
                continue

            reason = f"Matches goals, similarity score {float(1.0 - d):.2f}"
            recs.append(
                Recommendation(
                    exercise_name=str(row["exercise_name"]),
                    muscle_group=str(row["muscle_group"]),
                    difficulty=str(row["difficulty"]),
                    reason=reason,
                )
            )
        return recs[:top_k]
