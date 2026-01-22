"""
Redis subscriber for transcription progress updates.
Forwards messages from Redis pub/sub to WebSocket clients.
"""

import json
import asyncio
import redis
from typing import Optional
from loguru import logger

from app.core.config import settings
from app.utils.websocket_manager import websocket_manager


class RedisSubscriber:
    """Subscribes to Redis channels and forwards messages to WebSocket clients."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the Redis subscriber."""
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            # Subscribe to transcription, summarization, and ingestion progress channels
            self.pubsub.psubscribe("transcription_progress:*")
            self.pubsub.psubscribe("summarization_progress:*")
            self.pubsub.psubscribe("ingestion_progress:*")
            self.running = True
            logger.info("Redis subscriber started for progress forwarding (transcription, summarization, ingestion)")
            
            # Start listening in background
            self.task = asyncio.create_task(self._listen())
        except Exception as e:
            logger.error(f"Failed to start Redis subscriber: {e}")
            self.running = False
    
    async def stop(self):
        """Stop the Redis subscriber."""
        self.running = False
        if self.pubsub:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except Exception as e:
                logger.warning(f"Error closing pubsub: {e}")
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Redis subscriber stopped")
    
    async def _listen(self):
        """Listen for Redis messages and forward to WebSocket clients."""
        while self.running:
            try:
                # Get message from Redis (non-blocking with timeout)
                message = self.pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'pmessage':
                    # Parse the message
                    try:
                        data = json.loads(message['data'])
                        document_id = data.get('document_id')
                        message_type = data.get('type')
                        
                        if not document_id:
                            continue
                        # Determine which channel group
                        channel: str = message.get('channel', '') or message.get('pattern', '')
                        if isinstance(channel, bytes):
                            channel = channel.decode()

                        if channel.startswith('transcription_progress:') or channel == 'transcription_progress:*':
                            # Forward transcription messages
                            if message_type == 'progress':
                                progress = data.get('progress', {})
                                await websocket_manager.send_progress(document_id, progress)
                            elif message_type == 'complete':
                                result = data.get('result', {})
                                await websocket_manager.send_complete(document_id, result)
                            elif message_type == 'error':
                                error = data.get('error', 'Unknown error')
                                await websocket_manager.send_error(document_id, error)
                            elif message_type == 'status':
                                status = data.get('status', {})
                                await websocket_manager.send_status(document_id, status)
                            elif message_type == 'segment':
                                segment = data.get('segment', {})
                                await websocket_manager.send_segment(document_id, segment)
                        elif channel.startswith('summarization_progress:') or channel == 'summarization_progress:*':
                            # Forward summarization messages
                            if message_type == 'progress':
                                progress = data.get('progress', {})
                                await websocket_manager.send_summarization_progress(document_id, progress)
                            elif message_type == 'complete':
                                result = data.get('result', {})
                                await websocket_manager.send_summarization_complete(document_id, result)
                            elif message_type == 'error':
                                error = data.get('error', 'Unknown error')
                                await websocket_manager.send_summarization_error(document_id, error)
                            elif message_type == 'status':
                                status = data.get('status', {})
                                await websocket_manager.send_summarization_status(document_id, status)
                        elif channel.startswith('ingestion_progress:') or channel == 'ingestion_progress:*':
                            # Forward ingestion messages (admin sources)
                            if message_type == 'progress':
                                progress = data.get('progress', {})
                                await websocket_manager.send_ingestion_progress(document_id, progress)
                            elif message_type == 'complete':
                                result = data.get('result', {})
                                await websocket_manager.send_ingestion_complete(document_id, result)
                            elif message_type == 'error':
                                error = data.get('error', 'Unknown error')
                                await websocket_manager.send_ingestion_error(document_id, error)
                            elif message_type == 'status':
                                status = data.get('status', {})
                                await websocket_manager.send_ingestion_status(document_id, status)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Error forwarding Redis message to WebSocket: {e}", exc_info=True)
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(0.1)
                
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error in Redis subscriber: {e}", exc_info=True)
                await asyncio.sleep(1)


# Global Redis subscriber instance
redis_subscriber = RedisSubscriber()
