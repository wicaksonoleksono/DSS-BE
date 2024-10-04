class repo:
    def weighted_product_with_subcriteria(self, criteria, decision_matrix) -> any:
        # Buat list untuk menyimpan bobot, tipe sub-kriteria, dan nama sub-kriteria
        subcriteria_weights = []
        subcriteria_types = []
        sub_criteria_names = []

        # Loop melalui kriteria utama dan sub-kriteria
        for criterion in criteria:
            if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
                # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
                subcriteria_weights.append(criterion['weight'])
                subcriteria_types.append(criterion['type'])
                sub_criteria_names.append(criterion['name'])
            else:
                # Loop pada setiap sub-kriteria
                for subcriterion in criterion['subcriteria']:
                    # Hitung bobot aktual sub-kriteria berdasarkan bobot kriteria utama
                    actual_weight = criterion['weight'] * subcriterion['weight']
                    subcriteria_weights.append(actual_weight)
                    subcriteria_types.append(subcriterion['type'])
                    sub_criteria_names.append(subcriterion['name'])

        # Konversi bobot sub-kriteria menjadi numpy array
        subcriteria_weights = np.array(subcriteria_weights, dtype=float)

        # Buat list untuk menyimpan pesan error
        error_list = []

        # Periksa apakah bobot kriteria atau sub-kriteria berada di antara 1-5
        for weight, name in zip(subcriteria_weights, sub_criteria_names):
            if not (1 <= weight <= 5):
                error_list.append(f"Bobot sub-kriteria '{name}' tidak valid. Bobot harus berada pada skala 1-5 (Nilai saat ini: {weight}).")

        # Jika ada pesan error, gabungkan menjadi satu pesan dan raise ValueError
        if error_list:
            # Buat pesan error lengkap
            error_message = (
                "Validasi Weighted Product gagal karena kesalahan pada bobot kriteria:\n" +
                "\n".join(error_list)
            )
            raise ValueError(error_message)

        # Buat matriks keputusan untuk sub-kriteria
        sub_decision_matrix = np.zeros((len(decision_matrix), len(subcriteria_weights)))

        # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
        for i, alternative in enumerate(decision_matrix):
            for j, sub_name in enumerate(sub_criteria_names):
                if sub_name not in alternative['criteria_scores']:
                    raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative}'.")

                # Validasi bahwa nilai alternatif tidak boleh 0
                if alternative['criteria_scores'][sub_name] == 0:
                    raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

                sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]

        # Lakukan normalisasi bobot jika perlu (jika total bobot != 1)
        if not np.isclose(subcriteria_weights.sum(), 1.0):
            subcriteria_weights /= subcriteria_weights.sum()

        print("Subcriteria Weights (after normalization):", subcriteria_weights)
        print("Subcriteria Decision Matrix:\n", sub_decision_matrix)

        # Inisialisasi matriks perpangkatan
        powered_matrix = np.zeros_like(sub_decision_matrix, dtype=float)

        # Lakukan operasi perpangkatan berdasarkan jenis kriteria
        for i, criterion_type in enumerate(subcriteria_types):
            column = sub_decision_matrix[:, i]

            if criterion_type == "cost":
                if np.any(column == 0):
                    raise ValueError(f"Nilai nol ditemukan pada sub-kriteria cost '{sub_criteria_names[i]}', tidak bisa membagi dengan nol.")
                powered_column = np.power(1 / column, subcriteria_weights[i])
            elif criterion_type == "benefit":
                powered_column = np.power(column, subcriteria_weights[i])
            else:
                raise ValueError(f"Jenis kriteria '{criterion_type}' pada sub-kriteria '{sub_criteria_names[i]}' tidak dikenal.")

            powered_matrix[:, i] = powered_column

        print("Powered Matrix:\n", powered_matrix)
        # Kalikan semua elemen per baris untuk mendapatkan skor total setiap alternatif
        scores = powered_matrix.prod(axis=1)

        # Normalisasi skor (opsional, tergantung pada kebutuhan hasil akhir)
        if not np.isclose(scores.sum(), 1.0):
            scores /= scores.sum()

        print("Final Normalized Scores:", scores)

        # Buat dictionary hasil dengan format {Alternative name: score}
        results = {alternative['alternative']: score for alternative, score in zip(decision_matrix, scores)}

        # Simpan hasilnya tanpa menggunakan .tolist()
        self.save_results("weighted_product_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

        # Kembalikan hasil akhir dengan format yang baru
        return results


    def simple_additive_weighting_with_subcriteria(self, criteria, decision_matrix) -> any:
        # Buat list untuk menyimpan bobot, tipe sub-kriteria, dan nama sub-kriteria
        subcriteria_weights = []
        subcriteria_types = []
        sub_criteria_names = []

        # Loop melalui kriteria utama dan sub-kriteria untuk menghitung bobot aktual
        for criterion in criteria:
            if 'subcriteria' not in criterion or len(criterion['subcriteria']) == 0:
                # Jika tidak ada sub-kriteria, tambahkan kriteria utama langsung
                subcriteria_weights.append(criterion['weight'])
                subcriteria_types.append(criterion['type'])
                sub_criteria_names.append(criterion['name'])
            else:
                # Loop pada setiap sub-kriteria
                for subcriterion in criterion['subcriteria']:
                    # Hitung bobot aktual sub-kriteria berdasarkan bobot kriteria utama
                    actual_weight = criterion['weight'] * subcriterion['weight']
                    subcriteria_weights.append(actual_weight)
                    subcriteria_types.append(subcriterion['type'])
                    sub_criteria_names.append(subcriterion['name'])

        # Validasi bahwa total bobot kriteria (atau sub-kriteria) harus sama dengan 1
        if not np.isclose(sum(subcriteria_weights), 1.0):
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
                    if not (1 <= criterion['weight'] <= 5):
                        main_criteria_details[criterion['name']] = f"dengan nilai {criterion['weight']}"

            # Buat pesan error berdasarkan kesalahan yang ditemukan
            if main_criteria_details:
                error_message = (
                    "Total bobot kriteria pada SAW harus sama dengan 1."
                    "Hasil yang salah:<br/>" +
                    "<br/>".join([f"Kriteria '{key}' pada {detail}" for key, detail in main_criteria_details.items()])
                )
            else:
                error_message = "Total bobot kriteria pada SAW harus sama dengan 1, namun tidak ada kriteria yang terdeteksi kesalahan bobotnya."

            raise ValueError(error_message)

        # Konversi bobot sub-kriteria menjadi numpy array
        subcriteria_weights = np.array(subcriteria_weights, dtype=float)

        # Buat matriks keputusan untuk sub-kriteria
        sub_decision_matrix = np.zeros((len(decision_matrix), len(subcriteria_weights)))

        # Isi matriks keputusan untuk sub-kriteria berdasarkan nama sub-kriteria
        for i, alternative in enumerate(decision_matrix):
            for j, sub_name in enumerate(sub_criteria_names):
                if sub_name not in alternative['criteria_scores']:
                    raise ValueError(f"Nilai sub-kriteria '{sub_name}' hilang pada alternatif '{alternative}'.")

                # Validasi bahwa nilai alternatif tidak boleh 0
                if alternative['criteria_scores'][sub_name] == 0:
                    raise ValueError(f"Nilai alternatif untuk '{sub_name}' pada alternatif '{alternative['alternative']}' tidak boleh 0.")

        sub_decision_matrix[i, j] = alternative['criteria_scores'][sub_name]

        # Lakukan normalisasi seperti pada metode SAW
        normalized_matrix = np.zeros_like(sub_decision_matrix, dtype=float)

        # Normalisasi per kolom berdasarkan jenis kriteria
        for i, criterion_type in enumerate(subcriteria_types):
            column = sub_decision_matrix[:, i]

            if criterion_type == "cost":
                min_value = column.min()
                if min_value == 0:
                    raise ValueError(f"Minimum value untuk cost sub-kriteria '{sub_criteria_names[i]}' tidak boleh nol.")
                normalized_column = min_value / column
            elif criterion_type == "benefit":
                max_value = column.max()
                if max_value == 0:
                    raise ValueError(f"Maximum value untuk benefit sub-kriteria '{sub_criteria_names[i]}' tidak boleh nol.")
                normalized_column = column / max_value
            else:
                raise ValueError(f"Jenis kriteria '{criterion_type}' pada sub-kriteria '{sub_criteria_names[i]}' tidak dikenal.")

            normalized_matrix[:, i] = normalized_column

        # Kalikan matriks normalisasi dengan bobot sub-kriteria
        weighted_matrix = normalized_matrix * subcriteria_weights

        # Jumlahkan setiap baris untuk mendapatkan skor per alternatif
        scores = weighted_matrix.sum(axis=1)

        # Buat dictionary hasil dengan format {Alternative name: score}
        results = {alternative['alternative']: score for alternative, score in zip(decision_matrix, scores)}

        # Simpan hasilnya tanpa menggunakan .tolist()
        self.save_results("simple_additive_weighting_with_subcriteria", subcriteria_weights, sub_decision_matrix, results)

        return results
