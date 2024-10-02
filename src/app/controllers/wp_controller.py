from typing import Literal
from flask import Blueprint, request, jsonify
from flask.wrappers import Response
from app.models.calculation_model import CalculationModel
import numpy as np

wp_bp = Blueprint("wp_bp", __name__)

calculation_model = CalculationModel()


@wp_bp.route("/calculate", methods=["POST"])
def calculate_wp() -> tuple[Response, Literal[400]] | tuple[Response, Literal[200]]:
    data = request.json
    criteria_weights = np.array(data["criteria_weights"])
    decision_matrix = np.array(data["decision_matrix"])
    criteria_types = data.get("criteria_types")  # Added criteria_types

    if (
        decision_matrix.shape[1] != len(criteria_weights)
        or len(criteria_types) != decision_matrix.shape[1]
    ):
        return (
            jsonify(
                {
                    "message": "The number of criteria weights and criteria types must match the number of columns in the decision matrix."
                }
            ),
            400,
        )

    scores = calculation_model.weighted_product(
        criteria_weights, decision_matrix, criteria_types
    )

    return jsonify({"scores": scores.tolist()}), 200


@wp_bp.route("/save", methods=["POST"])
def save_wp_results() -> tuple[Response, Literal[201]]:
    data = request.json
    method_name = "weighted_product"

    calculation_model.save_results(
        method_name,
        np.array(data["criteria_weights"]),
        np.array(data["decision_matrix"]),
        np.array(data["scores"]),
    )

    return jsonify({"message": "Results saved successfully."}), 201


@wp_bp.route("/results", methods=["GET"])
def get_wp_results() -> tuple[Response, Literal[200]]:
    results = calculation_model.get_results()

    return jsonify({"results": results}), 200
