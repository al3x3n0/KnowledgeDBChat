"""
WebSocket connection manager for real-time updates.
"""

from typing import Dict, Set
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
    
    async def send_progress(self, document_id: str, progress: dict):
        """Send progress update to all connected clients for a document."""
        if document_id not in self.active_connections:
            return
        
        message = {
            "type": "transcription_progress",
            "document_id": document_id,
            "progress": progress
        }
        
        # Send to all connected clients
        disconnected = set()
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send progress to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, document_id)
    
    async def send_complete(self, document_id: str, result: dict):
        """Send completion message to all connected clients."""
        if document_id not in self.active_connections:
            return
        
        message = {
            "type": "transcription_complete",
            "document_id": document_id,
            "result": result
        }
        
        disconnected = set()
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send completion to WebSocket: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.disconnect(ws, document_id)
    
    async def send_error(self, document_id: str, error: str):
        """Send error message to all connected clients."""
        if document_id not in self.active_connections:
            return
        
        message = {
            "type": "transcription_error",
            "document_id": document_id,
            "error": error
        }
        
        disconnected = set()
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send error to WebSocket: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.disconnect(ws, document_id)

    async def send_segment(self, document_id: str, segment: dict):
        """Send a partial transcription segment to clients."""
        if document_id not in self.active_connections:
            return
        message = {
            "type": "transcription_segment",
            "document_id": document_id,
            "segment": segment,
        }
        disconnected = set()
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send segment to WebSocket: {e}")
                disconnected.add(websocket)
        for ws in disconnected:
            self.disconnect(ws, document_id)

    async def send_status(self, document_id: str, status: dict):
        """Send document status update (flags) to clients."""
        if document_id not in self.active_connections:
            return
        message = {
            "type": "document_status",
            "document_id": document_id,
            "status": status,
        }
        disconnected = set()
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send status to WebSocket: {e}")
                disconnected.add(websocket)
        for ws in disconnected:
            self.disconnect(ws, document_id)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
