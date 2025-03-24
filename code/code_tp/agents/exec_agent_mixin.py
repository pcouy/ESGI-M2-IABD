from pathlib import Path
import traceback

class PeriodicExec:
    def step(self, *args, **kwargs):
        if self.training_steps % 1000 == 0:
            rundir = Path("python_run_in_training")
            rundir.mkdir(parents=True, exist_ok=True, mode=0o770)
            ok_dir = rundir / "ok"
            ok_dir.mkdir(parents=True, exist_ok=True, mode=0o770)
            err_dir = rundir / "err"
            err_dir.mkdir(parents=True, exist_ok=True, mode=0o770)
            for script in rundir.glob("*.py"):
                try:
                    with open(script) as f:
                        exec(f.read())
                    script.rename(ok_dir / script.name)
                except Exception as e:
                    script.rename(err_dir / script.name)
                    with open(err_dir / f"{script.name}.err", "w") as f:
                        f.write(traceback.format_exc())
        return super().step(*args, **kwargs)
