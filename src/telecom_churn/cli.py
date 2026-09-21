"""Command-line entry point for the corrected analysis workflow."""

import argparse
import json
from pathlib import Path

from .data import load_telecom_data, profile_data
from .modeling import hypothetical_campaign, risk_deciles, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--target-deciles", type=int)
    parser.add_argument("--contact-cost", type=float)
    parser.add_argument("--retention-success-rate", type=float)
    parser.add_argument("--annual-customer-value", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_telecom_data(args.data_dir)
    result = run_experiment(
        frame,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    deciles = risk_deciles(result.test_predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.cv_results.to_csv(args.output_dir / "cv_results.csv", index=False)
    deciles.to_csv(args.output_dir / "risk_deciles.csv", index=False)

    summary: dict[str, object] = {
        "data_profile": profile_data(frame),
        "validation": {
            "model_selection": f"{args.cv_folds}-fold stratified CV on training partition only",
            "final_evaluation": "single untouched stratified holdout",
            "random_state": args.random_state,
            "test_size": args.test_size,
        },
        "cv_results": result.cv_results.to_dict(orient="records"),
        "test_metrics": result.test_metrics,
    }

    scenario_values = (
        args.target_deciles,
        args.contact_cost,
        args.retention_success_rate,
        args.annual_customer_value,
    )
    if any(value is not None for value in scenario_values):
        if any(value is None for value in scenario_values):
            raise SystemExit(
                "Campaign analysis requires --target-deciles, --contact-cost, "
                "--retention-success-rate, and --annual-customer-value together."
            )
        summary["campaign_scenario"] = hypothetical_campaign(
            deciles,
            targeted_deciles=args.target_deciles,
            contact_cost=args.contact_cost,
            retention_success_rate=args.retention_success_rate,
            annual_customer_value=args.annual_customer_value,
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}, cv_results.csv, and risk_deciles.csv")


if __name__ == "__main__":
    main()
