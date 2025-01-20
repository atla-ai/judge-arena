import os
from pymongo import MongoClient
from pymongo.database import Database
from utils import get_logger, Vote
from datetime import datetime, timedelta, timezone
from typing import List

logger = get_logger()


def create_db_connection() -> Database:
    print(os.getenv("MONGO_URI"))
    print(os.getenv("MONGO_DB"))
    db = MongoClient(os.getenv("MONGO_URI")).get_database(os.getenv("MONGO_DB"))
    return db


def add_vote(vote: Vote, db: Database) -> None:
    try:
        db.get_collection("votes").insert_one(vote.__dict__)
        logger.info("Vote added to database")
    except Exception as e:
        logger.error("Error adding vote to database", exc_info=True)


def get_votes(db: Database) -> List[Vote]:
    now = datetime.now(timezone.utc)
    # Round down to nearest 10-second interval
    current_interval = now.replace(microsecond=0, second=now.second - (now.second % 10))
    votes = list(db.get_collection("votes").find({"timestamp": {"$lte": current_interval.isoformat()}}))
    return votes

#     current_hour = now.replace(minute=0, second=0, microsecond=0)
#    votes = list(db.get_collection("votes").find({"timestamp": {"$lte": current_hour.isoformat()}}))