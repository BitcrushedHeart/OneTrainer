from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig


class _Result:
    def __init__(self, exited=0):
        self.exited = exited


class _FakeConn:
    def __init__(self):
        self.commands = []

    def run(self, cmd, **kwargs):
        self.commands.append(cmd)
        return _Result(0)


def _make_cloud(url="https://github.com/BitcrushedHeart/OneTrainer", branch="bitcrushed-blend"):
    # Bypass __init__ (which opens SSH); we only exercise command construction.
    cloud = LinuxCloud.__new__(LinuxCloud)
    config = TrainConfig.default_values()
    config.cloud.onetrainer_dir = "/workspace/OneTrainer"
    config.cloud.git_url = url
    config.cloud.git_branch = branch
    cloud.config = config
    cloud.connection = _FakeConn()
    return cloud


def test_install_clones_when_missing_and_repoints_when_present():
    cloud = _make_cloud()
    cloud._install_onetrainer(update=True)
    setup_cmd = cloud.connection.commands[0]

    # Missing-dir branch still clones.
    assert "git clone" in setup_cmd
    # Existing-dir branch repoints a pre-baked checkout onto the configured fork/branch.
    assert "git remote set-url origin https://github.com/BitcrushedHeart/OneTrainer" in setup_cmd
    assert "git fetch origin bitcrushed-blend" in setup_cmd
    assert "checkout -B bitcrushed-blend" in setup_cmd
    assert "git reset --hard origin/bitcrushed-blend" in setup_cmd
    # Must preserve the pre-baked venv (no destructive reclone).
    assert "rm -rf" not in setup_cmd


def test_install_respects_configured_url_and_branch():
    cloud = _make_cloud(url="https://github.com/Someone/Fork", branch="featureX")
    cloud._install_onetrainer(update=False)
    setup_cmd = cloud.connection.commands[0]

    assert "https://github.com/Someone/Fork" in setup_cmd
    assert "git fetch origin featureX" in setup_cmd
    assert "git reset --hard origin/featureX" in setup_cmd
