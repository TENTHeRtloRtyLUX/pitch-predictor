import argparse
from datetime import datetime, timezone


def current_year():
    return datetime.now(timezone.utc).year


def parse_explicit_seasons(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def build_season_range(start_year, end_year=None):
    resolved_end_year = current_year() if end_year is None else int(end_year)
    resolved_start_year = int(start_year)
    if resolved_start_year > resolved_end_year:
        raise ValueError(
            f"start_year {resolved_start_year} cannot be greater than end_year {resolved_end_year}."
        )
    return list(range(resolved_start_year, resolved_end_year + 1))


def resolve_seasons(explicit_seasons=None, start_year=None, end_year=None):
    if explicit_seasons and str(explicit_seasons).strip():
        seasons = parse_explicit_seasons(explicit_seasons)
        if not seasons:
            raise ValueError("Explicit seasons input was provided but no valid years were found.")
        return seasons

    if start_year is None:
        raise ValueError("You must provide either explicit seasons or a start_year.")

    return build_season_range(start_year=start_year, end_year=end_year)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve retraining seasons from either an explicit list or a start-year range."
    )
    parser.add_argument("--seasons", default=None, help="Explicit comma-separated seasons, e.g. 2023,2024,2025")
    parser.add_argument("--start-year", type=int, default=None, help="Inclusive starting year for season range.")
    parser.add_argument("--end-year", type=int, default=None, help="Inclusive ending year for season range.")
    args = parser.parse_args()

    seasons = resolve_seasons(
        explicit_seasons=args.seasons,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(",".join(str(season) for season in seasons))


if __name__ == "__main__":
    main()
