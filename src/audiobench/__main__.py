"""Allow running as: python -m audiobench"""

from audiobench.cli.app import cli
from audiobench.observatory.forwarding_subscriber import ForwardingSubscriber

# Install the forwarding subscriber so that every event emitted by this CLI
# process is batched and forwarded to the daemon's drain thread → journal.db.
# Without this call the subscriber file exists but is never in the execution
# path — the Observatory would only see daemon-side events.
ForwardingSubscriber().install()

if __name__ == "__main__":
    cli()
