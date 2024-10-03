from typing import Literal
from flask import Blueprint, request, jsonify
from flask.wrappers import Response
from app.models.calculation_model import CalculationModel
import numpy as np

saw_bp = Blueprint("saw_bp", __name__)
calculation_model = CalculationModel()


@saw_bp.route("/calculate", methods=["POST"])
def calculate_saw() -> tuple[Response, Literal[400]] | tuple[Response, Literal[200]]:
    data = request.json
    criteria_weights = np.array(data["criteria_weights"])
    decision_matrix = np.array(data["decision_matrix"])
    criteria_types = data.get("criteria_types")  # Added criteria_types

    if decision_matrix.shape[1] != len(criteria_weights):
        return (
            jsonify(
                {
                    "message": "The number of criteria weights must match the number of columns in the decision matrix."
                }
            ),
            400,
        )

    scores = calculation_model.simple_additive_weighting(
        criteria_weights, decision_matrix, criteria_types
    )

    return jsonify({"scores": scores.tolist()}), 200


@saw_bp.route("/save", methods=["POST"])
def save_saw_results() -> tuple[Response, Literal[201]]:
    data = request.json
    method_name = "simple_additive_weighting"

    calculation_model.save_results(
        method_name,
        np.array(data["criteria_weights"]),
        np.array(data["decision_matrix"]),
        np.array(data["scores"]),
    )

    return jsonify({"message": "Results saved successfully."}), 201


@saw_bp.route("/results", methods=["GET"])
def get_saw_results() -> tuple[Response, Literal[200]]:
    results = calculation_model.get_results()

    return jsonify({"results": results}), 200

@saw_bp.route("v2/calculate", methods=["POST"])
def calculate_saw_with_subcriteria() -> tuple[Response, Literal[400]] | tuple[Response, Literal[200]]:
    data = request.json
    criteria = data["criteria"]
    decision_matrix = data["decision_matrix"]

    try:
        scores = calculation_model.simple_additive_weighting_with_subcriteria(criteria, decision_matrix)
        return jsonify({"scores": scores}), 200
    except ValueError as e:
        return jsonify({"message": str(e)}), 400