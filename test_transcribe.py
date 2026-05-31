from audiobench.cli.repl.dispatch import dispatch_command
from audiobench.cli.repl.session import ReplSession
from audiobench.cli.app import cli
from audiobench.core.focused_entity import FocusedEntity

session = ReplSession(cli)
session.focus = FocusedEntity(type="file", id=1, label="test.mp3")
print("Args before dispatch:", ["transcribe"])
# dispatch_command(session, ["transcribe"]) # this will launch click and might block or error
args = session.auto_inject_id(["transcribe"])
print("Injected args:", args)
