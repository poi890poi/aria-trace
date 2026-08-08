"""Copy a COLMAP database and remove all query-to-query match edges."""

import argparse
import shutil
import sqlite3
from pathlib import Path


MAX_IMAGE_ID = 2147483647


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-list", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(args.input), str(args.output))

    query_names = set(args.query_list.read_text(encoding="utf-8").splitlines())
    with sqlite3.connect(str(args.output)) as connection:
        query_ids = {
            image_id
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
            if name in query_names
        }
        removed = {}
        for table in ("matches", "two_view_geometries"):
            pair_ids = [
                pair_id
                for (pair_id,) in connection.execute("SELECT pair_id FROM {}".format(table))
                if pair_id // MAX_IMAGE_ID in query_ids and pair_id % MAX_IMAGE_ID in query_ids
            ]
            connection.executemany(
                "DELETE FROM {} WHERE pair_id = ?".format(table),
                ((pair_id,) for pair_id in pair_ids),
            )
            removed[table] = len(pair_ids)
        connection.commit()

        remaining_geometries = connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
        ).fetchone()[0]
    print("query images: {}".format(len(query_ids)))
    print("removed raw match pairs: {}".format(removed["matches"]))
    print("removed verified geometry pairs: {}".format(removed["two_view_geometries"]))
    print("remaining nonempty verified geometries: {}".format(remaining_geometries))


if __name__ == "__main__":
    main()
