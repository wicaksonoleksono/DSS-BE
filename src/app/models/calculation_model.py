import numpy as np
from app.connection.connection import Connection
import json


class CalculationModel:
    def __init__(self) -> None:
        self.collection = Connection.get_collection("results")

   


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
        

    
    ###################################
    #### Model with Sub Criteria #####
    ###################################

    def save_results(
        self, method_name: str, criteria_weights, decision_matrix, scores
    ) -> None:
        """
        Save the results of the calculation in the database.
        """
        decision_matrix_str = json.dumps(decision_matrix.tolist())

        # Tentukan format scores berdasarkan tipe data yang diterima
        if isinstance(scores, dict):
            scores_data = scores 
        else:
            scores_data = scores.tolist()  

        data = {
            "method": method_name,
            "criteria_weights": criteria_weights.tolist(),
            "decision_matrix": decision_matrix_str,
            "scores": scores_data,  
        }

        self.collection.add(data)

    def weighted_product_with_subcriteria(self, criteria, decision_matrix) -> any:
        """
        Implementasi metode Weighted Product (WP) dengan dukungan sub-kriteria.

        :param criteria: List of criteria, each may have subcriteria.
        :param decision_matrix: List of alternatives with scores.
        :return: Dictionary mapping alternative names to their WP scores (scaled 1-5).
        """
        # Inisialisasi list untuk bobot, tipe sub-kriteria, dan nama sub-kriteria
        subcriteria_weights = []
        subcriteria_types = []
        subcriteria_names = []

        # Loop melalui kriteria utama dan sub-kriteria untuk menghitung bobot aktual
        for criterion in criteria:
            if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
                # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
                subcriteria_weights.append(criterion['weight'])
                subcriteria_types.append(criterion['type'])
                subcriteria_names.append(criterion['name'])
            else:
                # Jika ada sub-kriteria, normalisasi bobot sub-kriteria
                total_sub_weight = sum([sub['weight'] for sub in criterion['subcriteria']])
                if total_sub_weight == 0:
                    raise ValueError(f"Total bobot sub-kriteria untuk kriteria '{criterion['name']}' tidak boleh nol.")

                for subcriterion in criterion['subcriteria']:
                    # Normalisasi bobot sub-kriteria untuk setiap kriteria utama agar totalnya sesuai
                    normalized_sub_weight = (subcriterion['weight'] / total_sub_weight) * criterion['weight']
                    # Pastikan bobot sub-kriteria positif
                    if normalized_sub_weight <= 0:
                        raise ValueError(f"Bobot sub-kriteria '{subcriterion['name']}' setelah normalisasi ({normalized_sub_weight}) harus positif.")
                    subcriteria_weights.append(normalized_sub_weight)
                    subcriteria_types.append(subcriterion['type'])
                    subcriteria_names.append(subcriterion['name'])

        # Validasi bahwa semua bobot sub-kriteria adalah positif
        for weight, name in zip(subcriteria_weights, subcriteria_names):
            if weight <= 0:
                raise ValueError(f"Bobot sub-kriteria '{name}' tidak valid. Bobot harus positif (Nilai saat ini: {weight}).")

        # Konversi bobot sub-kriteria menjadi numpy array
        subcriteria_weights = np.array(subcriteria_weights, dtype=float)

        # Buat matriks keputusan untuk sub-kriteria
        num_alternatives = len(decision_matrix)
        num_subcriteria = len(subcriteria_weights)
        sub_decision_matrix = np.zeros((num_alternatives, num_subcriteria), dtype=float)

        # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
        for i, alternative in enumerate(decision_matrix):
            scores = alternative['criteria_scores']
            for j, sub_name in enumerate(subcriteria_names):
                if sub_name not in scores:
                    raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative['alternative']}'.")
                value = scores[sub_name]
                if value == 0:
                    raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")
                sub_decision_matrix[i, j] = value

        # Lakukan normalisasi seperti pada metode WP
        # Normalisasi per kolom berdasarkan jenis kriteria
        for j, (weight, c_type) in enumerate(zip(subcriteria_weights, subcriteria_types)):
            column = sub_decision_matrix[:, j]
            if c_type == "benefit":
                max_value = column.max()
                if max_value == 0:
                    raise ValueError(f"Nilai maksimum untuk benefit sub-kriteria '{subcriteria_names[j]}' tidak boleh nol.")
                normalized_column = column / max_value
            elif c_type == "cost":
                min_value = column.min()
                if min_value == 0:
                    raise ValueError(f"Nilai minimum untuk cost sub-kriteria '{subcriteria_names[j]}' tidak boleh nol.")
                normalized_column = min_value / column
            else:
                raise ValueError(f"Jenis kriteria '{c_type}' pada sub-kriteria '{subcriteria_names[j]}' tidak dikenal.")
            # Assign normalized column
            sub_decision_matrix[:, j] = normalized_column

        # Lakukan operasi perpangkatan berdasarkan bobot
        powered_matrix = np.power(sub_decision_matrix, subcriteria_weights)

        # Hitung skor WP sebagai produk dari semua kolom
        scores = powered_matrix.prod(axis=1)

        # Normalisasi skor akhir ke skala 1-5
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            # Jika semua skor sama, set semua skor ke 3
            scaled_scores = np.full_like(scores, 3.0)
        else:
            scaled_scores = ((scores - min_score) / (max_score - min_score)) * 4 + 1  # Skala 1-5

        # Buat dictionary hasil dengan format {Alternative name: score}
        results = {alternative['alternative']: round(score, 4) for alternative, score in zip(decision_matrix, scaled_scores)}

        # Debugging: Pastikan types adalah numpy arrays
        print(f"Type of criteria_weights: {type(subcriteria_weights)}")  # Should be <class 'numpy.ndarray'>
        print(f"Type of sub_decision_matrix: {type(sub_decision_matrix)}")  # Should be <class 'numpy.ndarray'>

        # Simpan hasilnya
        self.save_results("weighted_product_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

        # Kembalikan hasil akhir dengan format yang baru
        return results

    def simple_additive_weighting_with_subcriteria(self, criteria, decision_matrix) :
        """
        Implementasi metode Simple Additive Weighting (SAW) dengan dukungan sub-kriteria.

        :param criteria: List of criteria, each may have subcriteria.
        :param decision_matrix: List of alternatives with scores.
        :return: Dictionary mapping alternative names to their SAW scores.
        """
        # Inisialisasi list untuk bobot, tipe sub-kriteria, dan nama sub-kriteria
        subcriteria_weights = []
        subcriteria_types = []
        subcriteria_names = []

        # Loop melalui kriteria utama dan sub-kriteria untuk menghitung bobot aktual
        for criterion in criteria:
            if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
                # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
                subcriteria_weights.append(criterion['weight'])
                subcriteria_types.append(criterion['type'])
                subcriteria_names.append(criterion['name'])
            else:
                # Loop pada setiap sub-kriteria
                for subcriterion in criterion['subcriteria']:
                    # Hitung bobot aktual sub-kriteria berdasarkan bobot kriteria utama
                    actual_weight = criterion['weight'] * subcriterion['weight']
                    subcriteria_weights.append(actual_weight)
                    subcriteria_types.append(subcriterion['type'])
                    subcriteria_names.append(subcriterion['name'])

        # Validasi bahwa total bobot sub-kriteria (atau kriteria utama) harus sama dengan 1
        total_weight = sum(subcriteria_weights)
        if not np.isclose(total_weight, 1.0):
            # Kumpulkan informasi kriteria dan sub-kriteria yang salah
            main_criteria_details = {}
            for criterion in criteria:
                if 'subcriteria' in criterion and criterion['subcriteria']:
                    # Jika ada sub-kriteria, hitung total bobot sub-kriteria untuk kriteria ini
                    total_sub_weight = sum([sub['weight'] for sub in criterion['subcriteria']])
                    if not np.isclose(total_sub_weight, 1.0):
                        sub_names = ", ".join([sub['name'] for sub in criterion['subcriteria']])
                        main_criteria_details[criterion['name']] = f"{sub_names} dengan nilai {{{', '.join([str(sub['weight']) for sub in criterion['subcriteria']])}}}"
                else:
                    # Jika tidak ada sub-kriteria, langsung periksa bobot kriteria utama
                    if not np.isclose(criterion['weight'], 1.0):
                        main_criteria_details[criterion['name']] = f"dengan nilai {criterion['weight']}"

            # Buat pesan error berdasarkan kesalahan yang ditemukan
            if main_criteria_details:
                error_message = (
                    "Total bobot kriteria pada SAW harus sama dengan 1.<br/>" +
                    "<br/>".join([f"Kriteria '{key}' pada {detail}" for key, detail in main_criteria_details.items()])
                )
            else:
                error_message = "Total bobot kriteria pada SAW harus sama dengan 1, namun tidak ada kriteria yang terdeteksi kesalahan bobotnya."

            raise ValueError(error_message)

        # Konversi bobot sub-kriteria menjadi numpy array
        subcriteria_weights = np.array(subcriteria_weights, dtype=float)

        # Buat matriks keputusan untuk sub-kriteria
        num_alternatives = len(decision_matrix)
        num_subcriteria = len(subcriteria_weights)
        sub_decision_matrix = np.zeros((num_alternatives, num_subcriteria))

        # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
        for i, alternative in enumerate(decision_matrix):
            scores = alternative['criteria_scores']
            for j, sub_name in enumerate(subcriteria_names):
                if sub_name not in scores:
                    raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative['alternative']}'.")
                value = scores[sub_name]
                if value == 0:
                    raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")
                sub_decision_matrix[i, j] = value

        # Lakukan normalisasi seperti pada metode SAW
        # Normalisasi per kolom berdasarkan jenis kriteria
        normalized_matrix = np.zeros_like(sub_decision_matrix, dtype=float)

        for j, (weight, c_type) in enumerate(zip(subcriteria_weights, subcriteria_types)):
            column = sub_decision_matrix[:, j]
            if c_type == "benefit":
                max_value = column.max()
                if max_value == 0:
                    raise ValueError(f"Nilai maksimum untuk benefit sub-kriteria '{subcriteria_names[j]}' tidak boleh nol.")
                normalized_column = column / max_value
            elif c_type == "cost":
                min_value = column.min()
                if min_value == 0:
                    raise ValueError(f"Nilai minimum untuk cost sub-kriteria '{subcriteria_names[j]}' tidak boleh nol.")
                normalized_column = min_value / column
            else:
                raise ValueError(f"Jenis kriteria '{c_type}' pada sub-kriteria '{subcriteria_names[j]}' tidak dikenal.")

            normalized_matrix[:, j] = normalized_column

        # Kalikan matriks normalisasi dengan bobot sub-kriteria
        weighted_matrix = normalized_matrix * subcriteria_weights

        # Jumlahkan setiap baris untuk mendapatkan skor per alternatif
        scores = weighted_matrix.sum(axis=1)

        # Validasi bahwa skor akhir tidak melebihi 1
        if not np.all(scores <= 1.0):
            raise ValueError("Skor akhir SAW tidak boleh melebihi 1.")

        # Buat dictionary hasil dengan format {Alternative name: score}
        results = {alternative['alternative']: round(score, 4) for alternative, score in zip(decision_matrix, scores)}

        # Simpan hasilnya
        self.save_results("simple_additive_weighting_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

        # Kembalikan hasil akhir dengan format yang baru
        return results

    


   

    ###################################
    #### Model Non SUb kriteria   #####
    ###################################
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
    



# repo
 # def simple_additive_weighting_with_subcriteria(self, criteria, decision_matrix) -> any:
    #     # Buat list untuk menyimpan bobot, tipe sub-kriteria, dan nama sub-kriteria
    #     subcriteria_weights = []
    #     subcriteria_types = []
    #     sub_criteria_names = []

    #     # Loop melalui kriteria utama dan sub-kriteria untuk menghitung bobot aktual
    #     for criterion in criteria:
    #         if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
    #             # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
    #             subcriteria_weights.append(criterion['weight'])
    #             subcriteria_types.append(criterion['type'])
    #             sub_criteria_names.append(criterion['name'])
    #         else:
    #             # Loop pada setiap sub-kriteria
    #             for subcriterion in criterion['subcriteria']:
    #                 # Hitung bobot aktual sub-kriteria berdasarkan bobot kriteria utama
    #                 actual_weight = criterion['weight'] * subcriterion['weight']
    #                 subcriteria_weights.append(actual_weight)
    #                 subcriteria_types.append(subcriterion['type'])
    #                 sub_criteria_names.append(subcriterion['name'])

    #     for alternative in decision_matrix:
    #                 for sub_name, value in alternative['criteria_scores'].items():
    #                     if value == 0:
    #                         raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

    #     # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
    #     for i, alternative in enumerate(decision_matrix):
    #         for j, sub_name in enumerate(sub_criteria_names):
    #             if sub_name not in alternative['criteria_scores']:
    #                 raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative}'.")

    #             # Validasi bahwa nilai alternatif tidak boleh 0
    #             if alternative['criteria_scores'][sub_name] == 0:
    #                 raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

    #             sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]


    #     # Validasi bahwa total bobot kriteria (atau sub-kriteria) harus sama dengan 1
    #     if not np.isclose(sum(subcriteria_weights), 1.0):
    #         # Kumpulkan informasi kriteria dan sub-kriteria yang salah
    #         main_criteria_details = {}
    #         for criterion in criteria:
    #             if 'subcriteria' in criterion and criterion['subcriteria']:
    #                 # Jika ada sub-kriteria, hitung total bobot sub-kriteria untuk kriteria ini
    #                 total_sub_weight = sum([sub['weight'] for sub in criterion['subcriteria']])
    #                 if not np.isclose(total_sub_weight, 1.0):
    #                     sub_names = ", ".join([sub['name'] for sub in criterion['subcriteria']])
    #                     main_criteria_details[criterion['name']] = f"{sub_names} dengan nilai {{{', '.join([str(sub['weight']) for sub in criterion['subcriteria']])}}}"
    #             else:
    #                 # Jika tidak ada sub-kriteria, langsung periksa bobot kriteria utama
    #                 if not (1 <= criterion['weight'] <= 5):
    #                     main_criteria_details[criterion['name']] = f"dengan nilai {criterion['weight']}"


    #         # Buat pesan error berdasarkan kesalahan yang ditemukan
    #         if main_criteria_details:
    #             error_message = (
    #                 "Total bobot kriteria pada SAW harus sama dengan 1."
    #                 "Hasil yang salah:<br/>" +
    #                 "<br/>".join([f"Kriteria '{key}' pada {detail}" for key, detail in main_criteria_details.items()])
    #             )
    #         else:
    #             error_message = "Total bobot kriteria pada SAW harus sama dengan 1, namun tidak ada kriteria yang terdeteksi kesalahan bobotnya."

    #         raise ValueError(error_message)


    #     # Konversi bobot sub-kriteria menjadi numpy array
    #     subcriteria_weights = np.array(subcriteria_weights, dtype=float)

    #     # Buat matriks keputusan untuk sub-kriteria
    #     sub_decision_matrix = np.zeros((len(decision_matrix), len(subcriteria_weights)))

    #     # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
    #     for i, alternative in enumerate(decision_matrix):
    #         for j, sub_name in enumerate(sub_criteria_names):
    #             if sub_name not in alternative['criteria_scores']:
    #                 raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative}'.")

    #             sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]

    #     # Lakukan normalisasi seperti pada metode SAW
    #     normalized_matrix = np.zeros_like(sub_decision_matrix, dtype=float)

    #     # Normalisasi per kolom berdasarkan jenis kriteria
    #     for i, criterion_type in enumerate(subcriteria_types):
    #         column = sub_decision_matrix[:, i]

    #         if criterion_type == "cost":
    #             min_value = column.min()
    #             if min_value == 0:
    #                 raise ValueError(f"Minimum value untuk cost sub-kriteria '{sub_criteria_names[i]}' tidak boleh nol.")
    #             normalized_column = min_value / column
    #         elif criterion_type == "benefit":
    #             max_value = column.max()
    #             if max_value == 0:
    #                 raise ValueError(f"Maximum value untuk benefit sub-kriteria '{sub_criteria_names[i]}' tidak boleh nol.")
    #             normalized_column = column / max_value
    #         else:
    #             raise ValueError(f"Jenis kriteria '{criterion_type}' pada sub-kriteria '{sub_criteria_names[i]}' tidak dikenal.")
            
    #         normalized_matrix[:, i] = normalized_column

    #         # Kalikan matriks normalisasi dengan bobot sub-kriteria
    #         weighted_matrix = normalized_matrix * subcriteria_weights

    

    #         # Jumlahkan setiap baris untuk mendapatkan skor per alternatif
    #         scores = weighted_matrix.sum(axis=1)

    #         # Buat dictionary hasil dengan format {Alternative name: score}
    #         results = {alternative['alternative']: score for alternative, score in zip(decision_matrix, scores)}

    #         # Simpan hasilnya
    #         self.save_results("simple_additive_weighting_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

    #         # Kembalikan hasil akhir dengan format yang baru
    #         return results



    # wp repo
    #  def weighted_product_with_subcriteria(self, criteria, decision_matrix) -> any:
    #     # Buat list untuk menyimpan bobot, tipe sub-kriteria, dan nama sub-kriteria
    #     subcriteria_weights = []
    #     subcriteria_types = []
    #     sub_criteria_names = []

    #     # Loop melalui kriteria utama dan sub-kriteria
    #     for criterion in criteria:
    #         if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
    #             # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
    #             subcriteria_weights.append(criterion['weight'])
    #             subcriteria_types.append(criterion['type'])
    #             sub_criteria_names.append(criterion['name'])
    #         else:
    #             # Loop pada setiap sub-kriteria
    #             for subcriterion in criterion['subcriteria']:
    #                 # Hitung bobot aktual sub-kriteria berdasarkan bobot kriteria utama
    #                 actual_weight = criterion['weight'] * subcriterion['weight']
    #                 subcriteria_weights.append(actual_weight)
    #                 subcriteria_types.append(subcriterion['type'])
    #                 sub_criteria_names.append(subcriterion['name'])

    #     # Konversi bobot sub-kriteria menjadi numpy array
    #     subcriteria_weights = np.array(subcriteria_weights, dtype=float)

    #     # Validasi bahwa bobot total tidak mengandung nilai negatif atau nol
    #     if not np.all(subcriteria_weights > 0):
    #         raise ValueError("Bobot sub-kriteria tidak boleh kurang dari atau sama dengan nol.")
        

    #     # Validasi bahwa nilai alternatif berdasarkan kriteria tidak boleh 0
    #     for alternative in decision_matrix:
    #         for sub_name, value in alternative['criteria_scores'].items():
    #             if value == 0:
    #                 raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

    #     # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
    #     for i, alternative in enumerate(decision_matrix):
    #         for j, sub_name in enumerate(sub_criteria_names):
    #             if sub_name not in alternative['criteria_scores']:
    #                 raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative}'.")

    #             # Validasi bahwa nilai alternatif tidak boleh 0
    #             if alternative['criteria_scores'][sub_name] == 0:
    #                 raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

    #             sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]





    #     # Validasi bahwa bobot kriteria (utama dan sub-kriteria) harus pada skala 1-5
    #     # Buat list untuk menyimpan pesan error
    #     error_list = []

    #     # Periksa apakah bobot kriteria atau sub-kriteria berada di antara 1-5
    #     for weight, name in zip(subcriteria_weights, sub_criteria_names):
    #         if not (1 <= weight <= 5):
    #             error_list.append(f"Bobot sub-kriteria '{name}' tidak valid. Bobot harus berada pada skala 1-5 (Nilai saat ini: {weight}).")

    #     # Jika ada pesan error, gabungkan menjadi satu pesan dan raise ValueError
    #     if error_list:
    #         # Buat pesan error lengkap
    #         error_message = (
    #             "Validasi Weighted Product gagal karena kesalahan pada bobot kriteria:<br/>" +
    #             "<br/>".join(error_list)
    #         )
    #         raise ValueError(error_message)
    #     # Buat matriks keputusan untuk sub-kriteria
    #     sub_decision_matrix = np.zeros((len(decision_matrix), len(subcriteria_weights)))

    #     # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
    #     for i, alternative in enumerate(decision_matrix):
    #         for j, sub_name in enumerate(sub_criteria_names):
    #             sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]

    #     # Lakukan normalisasi bobot jika perlu (jika total bobot != 1)
    #     if not np.isclose(subcriteria_weights.sum(), 1.0):
    #         subcriteria_weights /= subcriteria_weights.sum()

    #     print("Subcriteria Weights (after normalization):", subcriteria_weights)
    #     print("Subcriteria Decision Matrix:\n", sub_decision_matrix)

    #     # Inisialisasi matriks perpangkatan
    #     powered_matrix = np.zeros_like(sub_decision_matrix, dtype=float)

    #     # Lakukan operasi perpangkatan berdasarkan jenis kriteria
    #     for i, criterion_type in enumerate(subcriteria_types):
    #         column = sub_decision_matrix[:, i]

    #         if criterion_type == "cost":
    #             if np.any(column == 0):
    #                 raise ValueError(f"Nilai nol ditemukan pada sub-kriteria cost '{sub_criteria_names[i]}', tidak bisa membagi dengan nol.")
    #             powered_column = np.power(1 / column, subcriteria_weights[i])
    #         elif criterion_type == "benefit":
    #             powered_column = np.power(column, subcriteria_weights[i])
    #         else:
    #             raise ValueError(f"Jenis kriteria '{criterion_type}' pada sub-kriteria '{sub_criteria_names[i]}' tidak dikenal.")

    #         powered_matrix[:, i] = powered_column

    #     print("Powered Matrix:\n", powered_matrix)

    #     # Kalikan semua elemen per baris untuk mendapatkan skor total setiap alternatif
    #     scores = powered_matrix.prod(axis=1)

    #     # Normalisasi skor (opsional, tergantung pada kebutuhan hasil akhir)
    #     if not np.isclose(scores.sum(), 1.0):
    #         scores /= scores.sum()

    #     print("Final Normalized Scores:", scores)

    #     # Buat dictionary hasil dengan format {Alternative name: score}
    #     results = {alternative['alternative']: score for alternative, score in zip(decision_matrix, scores)}

    #     # Simpan hasilnya 
    #     self.save_results("weighted_product_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

    #     # Kembalikan hasil akhir dengan format yang baru
    #     return results  