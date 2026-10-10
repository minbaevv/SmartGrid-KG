import argparse
import asyncio
from pathlib import Path
from app.db import init_db, save_reservoir, engine
from app.services.reservoir import parse_reservoir_csv

async def main():
    parser = argparse.ArgumentParser(description="Import verified reservoir CSV")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    points = parse_reservoir_csv(args.csv)
    await init_db()
    print(f"Imported {await save_reservoir(points)} observations")

if __name__ == "__main__":
    async def run():
        try:
            await main()
        finally:
            await engine.dispose()
    asyncio.run(run())
