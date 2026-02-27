import click


class DefaultGroup(click.Group):
    """Invokes a default subcommand if the subcommand is missing."""

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop("default_command", None)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args and self.default_command is not None:
            args.insert(0, self.default_command)
            return super().parse_args(ctx, args)

        # If we have arguments, check if the first one is a known command or an option.
        if (
            args
            and args[0] not in self.commands
            and not args[0].startswith("-")
            and self.default_command is not None
        ):
            args.insert(0, self.default_command)
        return super().parse_args(ctx, args)
