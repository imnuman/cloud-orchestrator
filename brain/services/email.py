"""
Email Service for sending transactional emails.

Supports multiple providers:
- SMTP (default, works with any SMTP server)
- SendGrid (recommended for production)
- AWS SES (enterprise option)
"""

import logging
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

from brain.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmailProvider(ABC):
    """Abstract base class for email providers."""

    @abstractmethod
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """Send an email. Returns True if successful."""
        pass


class SMTPProvider(EmailProvider):
    """SMTP email provider."""

    def __init__(
        self,
        host: str,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_email: str = "noreply@example.com",
        from_name: str = "GPU Cloud",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_email = from_email
        self.from_name = from_name

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            if text_content:
                msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            if self.use_tls:
                server = smtplib.SMTP(self.host, self.port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.host, self.port)

            if self.username and self.password:
                server.login(self.username, self.password)

            server.sendmail(self.from_email, to_email, msg.as_string())
            server.quit()

            logger.info(f"Email sent to {to_email}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False


class SendGridProvider(EmailProvider):
    """SendGrid email provider."""

    def __init__(
        self,
        api_key: str,
        from_email: str = "noreply@example.com",
        from_name: str = "GPU Cloud",
    ):
        self.api_key = api_key
        self.from_email = from_email
        self.from_name = from_name
        self.api_url = "https://api.sendgrid.com/v3/mail/send"

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        try:
            payload = {
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": self.from_email, "name": self.from_name},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_content}],
            }

            if text_content:
                payload["content"].insert(0, {"type": "text/plain", "value": text_content})

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code in (200, 202):
                logger.info(f"Email sent via SendGrid to {to_email}: {subject}")
                return True
            else:
                logger.error(f"SendGrid error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send email via SendGrid to {to_email}: {e}")
            return False


class ConsoleProvider(EmailProvider):
    """Console provider for development - just logs emails."""

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        logger.info(f"[DEV EMAIL] To: {to_email}")
        logger.info(f"[DEV EMAIL] Subject: {subject}")
        logger.info(f"[DEV EMAIL] Content: {text_content or html_content[:200]}...")
        print(f"\n{'='*60}")
        print(f"[DEV EMAIL] To: {to_email}")
        print(f"[DEV EMAIL] Subject: {subject}")
        print(f"[DEV EMAIL] Content:\n{text_content or html_content}")
        print(f"{'='*60}\n")
        return True


class EmailService:
    """
    Email service that uses configured provider.

    Usage:
        service = get_email_service()
        await service.send_verification_email(user.email, token)
    """

    def __init__(self, provider: EmailProvider):
        self.provider = provider
        self.app_name = settings.app_name
        self.base_url = settings.brain_public_url

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """Send an email using the configured provider."""
        return await self.provider.send_email(to_email, subject, html_content, text_content)

    async def send_verification_email(self, to_email: str, token: str) -> bool:
        """Send email verification link."""
        verify_url = f"{self.base_url}/api/v1/auth/verify-email?token={token}"

        subject = f"Verify your {self.app_name} account"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }}
                .button:hover {{ background: #5a6fd6; }}
                .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 20px; }}
                .code {{ background: #e9e9e9; padding: 10px 15px; border-radius: 4px; font-family: monospace; word-break: break-all; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{self.app_name}</h1>
                </div>
                <div class="content">
                    <h2>Verify your email address</h2>
                    <p>Thanks for signing up! Please verify your email address to complete your registration and start using GPU resources.</p>
                    <p style="text-align: center;">
                        <a href="{verify_url}" class="button">Verify Email Address</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p class="code">{verify_url}</p>
                    <p>This link will expire in 24 hours.</p>
                    <p>If you didn't create an account, you can safely ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; {self.app_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Verify your {self.app_name} account

        Thanks for signing up! Please verify your email address by clicking the link below:

        {verify_url}

        This link will expire in 24 hours.

        If you didn't create an account, you can safely ignore this email.
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_password_reset_email(self, to_email: str, token: str) -> bool:
        """Send password reset link."""
        reset_url = f"{self.base_url}/reset-password?token={token}"

        subject = f"Reset your {self.app_name} password"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }}
                .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 20px; }}
                .code {{ background: #e9e9e9; padding: 10px 15px; border-radius: 4px; font-family: monospace; word-break: break-all; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{self.app_name}</h1>
                </div>
                <div class="content">
                    <h2>Reset your password</h2>
                    <p>We received a request to reset your password. Click the button below to create a new password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_url}" class="button">Reset Password</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p class="code">{reset_url}</p>
                    <p>This link will expire in 1 hour.</p>
                    <p>If you didn't request a password reset, you can safely ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; {self.app_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Reset your {self.app_name} password

        We received a request to reset your password. Click the link below to create a new password:

        {reset_url}

        This link will expire in 1 hour.

        If you didn't request a password reset, you can safely ignore this email.
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_welcome_email(self, to_email: str, name: Optional[str] = None) -> bool:
        """Send welcome email after verification."""
        subject = f"Welcome to {self.app_name}!"
        display_name = name or "there"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }}
                .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 20px; }}
                .feature {{ padding: 15px; margin: 10px 0; background: white; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to {self.app_name}!</h1>
                </div>
                <div class="content">
                    <h2>Hi {display_name}!</h2>
                    <p>Your email has been verified and your account is now active. Here's what you can do:</p>

                    <div class="feature">
                        <strong>Deploy AI Models</strong>
                        <p>One-click deployment of Llama, Mistral, SDXL, Whisper, and more.</p>
                    </div>

                    <div class="feature">
                        <strong>Rent GPUs On-Demand</strong>
                        <p>Access RTX 4090s, A100s, and H100s at competitive prices.</p>
                    </div>

                    <div class="feature">
                        <strong>Pay Per Hour</strong>
                        <p>Simple billing - pay only for what you use, no token counting.</p>
                    </div>

                    <p style="text-align: center;">
                        <a href="{self.base_url}" class="button">Get Started</a>
                    </p>

                    <p>Need help? Reply to this email or check our documentation.</p>
                </div>
                <div class="footer">
                    <p>&copy; {self.app_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Welcome to {self.app_name}!

        Hi {display_name}!

        Your email has been verified and your account is now active. Here's what you can do:

        - Deploy AI Models: One-click deployment of Llama, Mistral, SDXL, Whisper, and more.
        - Rent GPUs On-Demand: Access RTX 4090s, A100s, and H100s at competitive prices.
        - Pay Per Hour: Simple billing - pay only for what you use, no token counting.

        Get started at: {self.base_url}

        Need help? Reply to this email or check our documentation.
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_low_balance_warning(
        self, to_email: str, balance: float, threshold: float, name: Optional[str] = None
    ) -> bool:
        """Send low balance warning email."""
        subject = f"[{self.app_name}] Low balance warning"
        display_name = name or "there"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #f59e0b; color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }}
                .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 20px; }}
                .balance {{ font-size: 36px; font-weight: bold; color: #f59e0b; text-align: center; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Low Balance Warning</h1>
                </div>
                <div class="content">
                    <h2>Hi {display_name},</h2>
                    <p>Your {self.app_name} account balance is running low:</p>

                    <div class="balance">${balance:.2f}</div>

                    <p>Your running pods may be stopped if your balance reaches $0. Add funds to keep your services running.</p>

                    <p style="text-align: center;">
                        <a href="{self.base_url}/billing" class="button">Add Funds</a>
                    </p>

                    <p><strong>Tip:</strong> Enable auto-refill to automatically top up your balance when it drops below ${threshold:.2f}.</p>
                </div>
                <div class="footer">
                    <p>&copy; {self.app_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Low Balance Warning

        Hi {display_name},

        Your {self.app_name} account balance is running low: ${balance:.2f}

        Your running pods may be stopped if your balance reaches $0. Add funds to keep your services running.

        Add funds at: {self.base_url}/billing

        Tip: Enable auto-refill to automatically top up your balance when it drops below ${threshold:.2f}.
        """

        return await self.send_email(to_email, subject, html_content, text_content)


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get the configured email service instance."""
    global _email_service

    if _email_service is None:
        # Determine provider based on settings
        if settings.email_provider == "sendgrid" and settings.sendgrid_api_key:
            provider = SendGridProvider(
                api_key=settings.sendgrid_api_key,
                from_email=settings.email_from_address,
                from_name=settings.email_from_name,
            )
            logger.info("Email service initialized with SendGrid provider")
        elif settings.email_provider == "smtp" and settings.smtp_host:
            provider = SMTPProvider(
                host=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                use_tls=settings.smtp_use_tls,
                from_email=settings.email_from_address,
                from_name=settings.email_from_name,
            )
            logger.info(f"Email service initialized with SMTP provider ({settings.smtp_host})")
        else:
            # Default to console provider for development
            provider = ConsoleProvider()
            logger.info("Email service initialized with Console provider (dev mode)")

        _email_service = EmailService(provider)

    return _email_service
