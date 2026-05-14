from sneeze.commandinvariant import InvariantAwareCommand


class TpnPluginInfo(InvariantAwareCommand):
    """Show basic information about this plugin."""

    def run(self):
        self._out("sneeze plugin: tpn")
