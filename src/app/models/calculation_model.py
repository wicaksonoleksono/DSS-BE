import numpy as np
from app.connection.connection import Connection
import json


class CalculationModel:
    def __init__(self) -> None:
        self.collection = Connection.get_collection("results")

    def simple_additive_weighting(
        self, criteria_weights, decision_matrix, criteria_types
    ) -> any:
        # Ubah data JSON ke numpy array
        criteria_weights = np.array(criteria_weights, dtype=float)
        decision_matrix = np.array(decision_matrix, dtype=float)

        print("Criteria Weights:", criteria_weights)
        print("Decision Matrix:\n", decision_matrix)
        print("Criteria Types:", criteria_types)
        # Cek apakah jumlah bobot kriteria sama dengan jumlah kolom pada matriks keputusan
        if len(criteria_weights) != decision_matrix.shape[1]:
            raise ValueError(
                "The number of criteria weights must match the number of columns in the decision matrix."
            )
        # Inisialisasi matriks normalisasi dengan bentuk yang sama seperti matriks keputusan
        normalized_matrix = np.zeros_like(decision_matrix, dtype=float)

        # Loop melalui jenis kriteria untuk normalisasi
        for i, criterion_type in enumerate(criteria_types):
            column = decision_matrix[:, i]
            # Jika kriteria 'cost', normalisasi dengan min_value / column
            if criterion_type == "cost":
                min_value = column.min()
                if min_value == 0:
                    raise ValueError(
                        f"Minimum value for cost criterion at index {i} is zero, cannot divide by zero."
                    )
                normalized_column = min_value / column
            # Jika kriteria 'benefit', normalisasi dengan column / max_value
            elif criterion_type == "benefit":
                max_value = column.max()
                if max_value == 0:
                    raise ValueError(
                        f"Maximum value for benefit criterion at index {i} is zero, cannot divide by zero."
                    )
                normalized_column = column / max_value
            else:
                raise ValueError(
                    f"Unknown criterion type '{criterion_type}' at index {i}."
                )
            # Simpan hasil normalisasi ke dalam matriks normalisasi

            normalized_matrix[:, i] = normalized_column

            print(
                f"Normalized values for criterion {i} ({criterion_type}):",
                normalized_column,
            )

        print("Normalized Matrix:\n", normalized_matrix)
        # Kalikan matriks normalisasi dengan bobot kriteria
        weighted_matrix = normalized_matrix * criteria_weights

        print("Weighted Matrix:\n", weighted_matrix)
        # Jumlahkan setiap baris untuk mendapatkan skor per alternatif
        scores = weighted_matrix.sum(axis=1)

        print("Scores:", scores)

        self.save_results(
            "simple_additive_weighting", criteria_weights, decision_matrix, scores
        )
        # Kembalikan skor akhir
        return scores

    def weighted_product(
        self, criteria_weights, decision_matrix, criteria_types
    ) -> any:
        # Ubah data JSON ke numpy array
        criteria_weights = np.array(criteria_weights, dtype=float)
        decision_matrix = np.array(decision_matrix, dtype=float)

        print("Criteria Weights (before normalization):", criteria_weights)
        print("Decision Matrix:\n", decision_matrix)
        print("Criteria Types:", criteria_types)
        # Cek apakah jumlah bobot kriteria sama dengan jumlah kolom pada matriks keputusan
        if len(criteria_weights) != decision_matrix.shape[1]:
            raise ValueError(
                "The number of criteria weights must match the number of columns in the decision matrix."
            )
        # Normalisasi bobot kriteria
        criteria_weights /= criteria_weights.sum()
        print("Criteria Weights (after normalization):", criteria_weights)
        # Inisialisasi matriks perpangkatan
        powered_matrix = np.zeros_like(decision_matrix, dtype=float)
        #
        for i, criterion_type in enumerate(criteria_types):
            column = decision_matrix[:, i]
            # Lakukan operasi per kolom berdasarkan jenis kriteria

            if criterion_type == "cost":
                # Jika kriteria 'cost', pangkatkan (1 / column) dengan bobot kriteria
                if np.any(column == 0):
                    raise ValueError(
                        f"Zero value found in cost criterion at index {i}, cannot divide by zero."
                    )
                powered_column = np.power(1 / column, criteria_weights[i])
            elif criterion_type == "benefit":
                # Jika kriteria 'benefit', pangkatkan column dengan bobot kriteria
                powered_column = np.power(column, criteria_weights[i])
            else:
                raise ValueError(
                    f"Unknown criterion type '{criterion_type}' at index {i}."
                )
            # Simpan hasil perpangkatan ke dalam matriks perpangkatan
            powered_matrix[:, i] = powered_column

            print(
                f"Powered values for criterion {i} ({criterion_type}):", powered_column
            )

        print("Powered Matrix:\n", powered_matrix)
        # Kalikan semua elemen per baris untuk mendapatkan skor
        scores = powered_matrix.prod(axis=1)
        # Normalisasi skor
        scores /= scores.sum()
        print("Normalized Scores:", scores)

        self.save_results("weighted_product", criteria_weights, decision_matrix, scores)

        return scores

    def save_results(
        self, method_name, criteria_weights, decision_matrix, scores
    ) -> None:
        """
        Save the results of the calculation in the database.
        """
        decision_matrix_str = json.dumps(decision_matrix.tolist())

        data = {
            "method": method_name,
            "criteria_weights": criteria_weights.tolist(),
            "decision_matrix": decision_matrix_str,
            "scores": scores.tolist(),
        }

        self.collection.add(data)

    def get_results(self):
        """
        Retrieve results from the database.
        """
        docs = self.collection.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if "decision_matrix" in data:
                try:
                    data["decision_matrix"] = json.loads(data["decision_matrix"])
                except json.JSONDecodeError:
                    data["decision_matrix"] = None
            else:
                data["decision_matrix"] = None
            results.append(data)
        return results
