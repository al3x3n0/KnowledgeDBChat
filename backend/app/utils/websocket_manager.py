"""
WebSocket connection manager for real-time updates.
"""

from typing import Dict, List, Set
from fastapi import WebSocket
from loguru import logger
import json


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # Map of document_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, document_id: str):
        """Connect a WebSocket for a specific document."""
        await websocket.accept()

        if document_id not in self.active_connections:
            self.active_connections[document_id] = set()

        self.active_connections[document_id].add(websocket)
        logger.info(f"WebSocket connected for document {document_id}. Total connections: {len(self.active_connections[document_id])}")

    def disconnect(self, websocket: WebSocket, document_id: str):
        """Disconnect a WebSocket."""
        if document_id in self.active_connections:
            self.active_connections[document_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[document_id]:
                del self.active_connections[document_id]

            logger.info(f"WebSocket disconnected for document {document_id}")

    def _get_connections(self, key: str) -> List[WebSocket]:
        """Get a snapshot of connections for a key to avoid iteration issues."""
        if key not in self.active_connections:
            return []
        return list(self.active_connections[key])

    async def _broadcast(self, key: str, message: dict) -> None:
        """Broadcast a message to all connections for a key, handling disconnections safely."""
        connections = self._get_connections(key)
        if not connections:
            return

        disconnected: List[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(websocket)

        # Remove disconnected websockets after iteration
        for ws in disconnected:
            self.disconnect(ws, key)
    
    async def send_progress(self, document_id: str, progress: dict):
        """Send progress update to all connected clients for a document."""
        await self._broadcast(document_id, {
            "type": "transcription_progress",
            "document_id": document_id,
            "progress": progress
        })

    async def send_complete(self, document_id: str, result: dict):
        """Send completion message to all connected clients."""
        await self._broadcast(document_id, {
            "type": "transcription_complete",
            "document_id": document_id,
            "result": result
        })

    async def send_error(self, document_id: str, error: str):
        """Send error message to all connected clients."""
        await self._broadcast(document_id, {
            "type": "transcription_error",
            "document_id": document_id,
            "error": error
        })

    async def send_segment(self, document_id: str, segment: dict):
        """Send a partial transcription segment to clients."""
        await self._broadcast(document_id, {
            "type": "transcription_segment",
            "document_id": document_id,
            "segment": segment,
        })

    async def send_status(self, document_id: str, status: dict):
        """Send document status update (flags) to clients."""
        await self._broadcast(document_id, {
            "type": "document_status",
            "document_id": document_id,
            "status": status,
        })

    # Summarization-specific events
    async def send_summarization_progress(self, document_id: str, progress: dict):
        """Send summarization progress to clients."""
        await self._broadcast(document_id, {
            "type": "summarization_progress",
            "document_id": document_id,
            "progress": progress,
        })

    async def send_summarization_complete(self, document_id: str, result: dict):
        """Send summarization completion to clients."""
        await self._broadcast(document_id, {
            "type": "summarization_complete",
            "document_id": document_id,
            "result": result,
        })

    async def send_summarization_error(self, document_id: str, error: str):
        """Send summarization error to clients."""
        await self._broadcast(document_id, {
            "type": "summarization_error",
            "document_id": document_id,
            "error": error,
        })

    async def send_summarization_status(self, document_id: str, status: dict):
        """Send summarization status to clients."""
        await self._broadcast(document_id, {
            "type": "summarization_status",
            "document_id": document_id,
            "status": status,
        })

    # Ingestion-specific events (source_id keyed)
    async def send_ingestion_progress(self, source_id: str, progress: dict):
        """Send ingestion progress to clients."""
        await self._broadcast(source_id, {
            "type": "ingestion_progress",
            "source_id": source_id,
            "progress": progress,
        })

    async def send_ingestion_complete(self, source_id: str, result: dict):
        """Send ingestion completion to clients."""
        await self._broadcast(source_id, {
            "type": "ingestion_complete",
            "source_id": source_id,
            "result": result,
        })

    async def send_ingestion_error(self, source_id: str, error: str):
        """Send ingestion error to clients."""
        await self._broadcast(source_id, {
            "type": "ingestion_error",
            "source_id": source_id,
            "error": error,
        })

    async def send_ingestion_status(self, source_id: str, status: dict):
        """Send ingestion status to clients."""
        await self._broadcast(source_id, {
            "type": "ingestion_status",
            "source_id": source_id,
            "status": status,
        })


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
