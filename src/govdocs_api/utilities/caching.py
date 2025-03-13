
from redis_om import get_redis_connection
import json
from typing import Optional, List
s
# Redis caching using redis-om

redis_client = get_redis_connection(
    host='localhost',
    port=6379,
    decode_responses=True
)

def cache_key(model: str, image_path: str, **kwargs) -> str:
  """Generate a unique cache key for the given model and image path."""
  params = json.dumps(kwargs, sort_keys=True)
  return f"{model}:{image_path}:{params}"


def get_cached_result(key: str) -> Optional[str]:
  """Get the cached result for the given key."""
  return redis_client.get(key)


def set_cached_result(key: str, result: str, expire_time: int = 60000) -> None:
  """Store the result in the Redis cache with the given expiration key."""
  redis_client.setex(key, expire_time, result)