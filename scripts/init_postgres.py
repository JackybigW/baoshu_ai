import asyncio
from pathlib import Path
import sys

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(find_dotenv(), override=True)

from db import db_store


async def main() -> None:
    await db_store.connect()
    await db_store.close()


if __name__ == "__main__":
    asyncio.run(main())
