"""
Model Proxy Service.

Routes user requests to their deployed models with:
- OpenAI-compatible API endpoints
- Authentication via deployment API keys
- Request/response logging for usage tracking
- WebSocket support for real-time streaming
"""

import logging
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.models.model_catalog import (
    Deployment,
    DeploymentStatus,
    ModelTemplate,
    ModelUsageLog,
)
from brain.models.pod import Pod

logger = logging.getLogger(__name__)

# Timeout settings for model requests
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=300.0,  # Long read timeout for inference
    write=30.0,
    pool=10.0,
)

# Streaming timeout (longer for streaming responses)
STREAMING_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=600.0,  # 10 minutes for streaming
    write=30.0,
    pool=10.0,
)


class ModelProxyService:
    """
    Service for proxying requests to deployed models.

    Handles:
    - Request authentication via deployment API keys
    - Routing to correct pod endpoints
    - OpenAI-compatible API translation
    - Usage logging and metrics
    - Streaming responses
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_deployment_by_api_key(
        self,
        api_key: str,
        db: AsyncSession,
    ) -> Optional[Deployment]:
        """
        Get deployment by API key.

        Args:
            api_key: Deployment API key
            db: Database session

        Returns:
            Deployment or None
        """
        result = await db.execute(
            select(Deployment).where(Deployment.api_key == api_key)
        )
        return result.scalar_one_or_none()

    async def get_deployment_endpoint(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> Optional[str]:
        """
        Get the internal endpoint URL for a deployment.

        Args:
            deployment: Deployment to get endpoint for
            db: Database session

        Returns:
            Internal endpoint URL or None
        """
        if deployment.status != DeploymentStatus.READY:
            return None

        if not deployment.pod_id:
            return None

        # Get pod to find IP/port
        result = await db.execute(select(Pod).where(Pod.id == deployment.pod_id))
        pod = result.scalar_one_or_none()

        if not pod:
            return None

        # Get model template for port info
        result = await db.execute(
            select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
        )
        template = result.scalar_one_or_none()

        if not template:
            return None

        # Build endpoint URL
        # In production, this would be the pod's internal IP/port
        # For now, use the deployment's api_endpoint if set
        if deployment.api_endpoint:
            return deployment.api_endpoint

        # Default endpoint format
        default_port = list(template.default_ports.keys())[0] if template.default_ports else "8000"
        return f"http://{pod.id}:{default_port}"

    async def proxy_request(
        self,
        deployment: Deployment,
        path: str,
        method: str,
        body: Optional[dict],
        headers: dict,
        db: AsyncSession,
    ) -> tuple[int, dict, Any]:
        """
        Proxy a request to the deployed model.

        Args:
            deployment: Target deployment
            path: Request path (e.g., "/v1/chat/completions")
            method: HTTP method
            body: Request body
            headers: Request headers
            db: Database session

        Returns:
            Tuple of (status_code, response_headers, response_body)
        """
        start_time = datetime.utcnow()
        input_tokens = 0
        output_tokens = 0

        try:
            endpoint = await self.get_deployment_endpoint(deployment, db)
            if not endpoint:
                return 503, {}, {"error": "Deployment endpoint not available"}

            # Build target URL
            url = urljoin(endpoint, path.lstrip("/"))

            # Forward headers (filter out sensitive ones)
            forward_headers = {
                k: v
                for k, v in headers.items()
                if k.lower()
                not in ["host", "authorization", "x-api-key", "content-length"]
            }

            # Make request
            response = await self.client.request(
                method=method,
                url=url,
                json=body if body else None,
                headers=forward_headers,
            )

            response_body = response.json() if response.content else {}

            # Extract token counts from response (OpenAI format)
            if "usage" in response_body:
                input_tokens = response_body["usage"].get("prompt_tokens", 0)
                output_tokens = response_body["usage"].get("completion_tokens", 0)

            # Log usage
            await self._log_usage(
                deployment=deployment,
                path=path,
                method=method,
                status_code=response.status_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                db=db,
            )

            return response.status_code, dict(response.headers), response_body

        except httpx.TimeoutException:
            await self._log_usage(
                deployment=deployment,
                path=path,
                method=method,
                status_code=504,
                input_tokens=0,
                output_tokens=0,
                latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                db=db,
                error="Request timeout",
            )
            return 504, {}, {"error": "Request timeout"}

        except httpx.RequestError as e:
            await self._log_usage(
                deployment=deployment,
                path=path,
                method=method,
                status_code=502,
                input_tokens=0,
                output_tokens=0,
                latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                db=db,
                error=str(e),
            )
            return 502, {}, {"error": f"Failed to reach model: {str(e)}"}

        except Exception as e:
            logger.error(f"Proxy error for deployment {deployment.id}: {e}")
            await self._log_usage(
                deployment=deployment,
                path=path,
                method=method,
                status_code=500,
                input_tokens=0,
                output_tokens=0,
                latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                db=db,
                error=str(e),
            )
            return 500, {}, {"error": "Internal proxy error"}

    async def proxy_streaming_request(
        self,
        deployment: Deployment,
        path: str,
        body: Optional[dict],
        headers: dict,
        db: AsyncSession,
    ):
        """
        Proxy a streaming request to the deployed model.

        Args:
            deployment: Target deployment
            path: Request path
            body: Request body
            headers: Request headers
            db: Database session

        Yields:
            Streaming response chunks
        """
        start_time = datetime.utcnow()
        total_tokens = 0

        try:
            endpoint = await self.get_deployment_endpoint(deployment, db)
            if not endpoint:
                yield b'data: {"error": "Deployment endpoint not available"}\n\n'
                return

            url = urljoin(endpoint, path.lstrip("/"))

            forward_headers = {
                k: v
                for k, v in headers.items()
                if k.lower()
                not in ["host", "authorization", "x-api-key", "content-length"]
            }

            # Use streaming client
            async with httpx.AsyncClient(timeout=STREAMING_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=forward_headers,
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        # Count chunks for rough token estimation
                        total_tokens += len(chunk) // 4  # Rough estimate

            # Log usage after streaming completes
            await self._log_usage(
                deployment=deployment,
                path=path,
                method="POST",
                status_code=200,
                input_tokens=0,
                output_tokens=total_tokens,
                latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                db=db,
            )

        except Exception as e:
            logger.error(f"Streaming proxy error for deployment {deployment.id}: {e}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()

    async def _log_usage(
        self,
        deployment: Deployment,
        path: str,
        method: str,
        status_code: int,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        db: AsyncSession,
        error: Optional[str] = None,
    ) -> None:
        """Log API usage for metrics and analytics."""
        try:
            log = ModelUsageLog(
                deployment_id=deployment.id,
                user_id=deployment.user_id,
                endpoint=path,
                method=method,
                status_code=status_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error_message=error,
            )
            db.add(log)
            await db.flush()
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")

    async def health_check(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> dict:
        """
        Check health of a deployed model.

        Args:
            deployment: Deployment to check
            db: Database session

        Returns:
            Health status dict
        """
        try:
            endpoint = await self.get_deployment_endpoint(deployment, db)
            if not endpoint:
                return {"status": "unavailable", "error": "No endpoint"}

            # Get template for health check URL
            result = await db.execute(
                select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
            )
            template = result.scalar_one_or_none()

            health_url = template.health_check_url if template else "/health"
            url = urljoin(endpoint, health_url.lstrip("/"))

            response = await self.client.get(url, timeout=5.0)

            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global singleton
_model_proxy_service: Optional[ModelProxyService] = None


def get_model_proxy_service() -> ModelProxyService:
    """Get the global ModelProxyService instance."""
    global _model_proxy_service
    if _model_proxy_service is None:
        _model_proxy_service = ModelProxyService()
    return _model_proxy_service
