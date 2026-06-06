from unittest.mock import MagicMock, patch

from audiobench.daemon.client import DaemonClient
from audiobench.daemon.factory import get_daemon_client
from audiobench.daemon.local_client import LocalRetrievalClient


@patch("audiobench.daemon.factory._is_socket_alive")
@patch("audiobench.daemon.factory.Path.exists")
def test_factory_returns_daemon_if_alive(mock_exists: MagicMock, mock_is_alive: MagicMock) -> None:
    """If the socket exists and is alive, return DaemonClient immediately."""
    mock_exists.return_value = True
    mock_is_alive.return_value = True

    with patch("audiobench.daemon.factory.DaemonClient") as mock_client_cls:
        mock_client = MagicMock(spec=DaemonClient)
        mock_client_cls.return_value = mock_client

        client = get_daemon_client()
        assert client is mock_client


@patch("audiobench.daemon.factory._start_daemon")
@patch("audiobench.daemon.factory.Path.exists")
def test_factory_fallback_on_start_failure(mock_exists: MagicMock, mock_start: MagicMock) -> None:
    """If the socket is dead and auto-start fails, return LocalRetrievalClient."""
    mock_exists.return_value = False
    mock_start.return_value = False  # Auto-start failed / timed out

    with patch("audiobench.daemon.factory.LocalRetrievalClient") as mock_local_cls:
        mock_local = MagicMock(spec=LocalRetrievalClient)
        mock_local_cls.return_value = mock_local

        client = get_daemon_client()
        assert client is mock_local
