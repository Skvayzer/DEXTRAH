"""Guard the diagnostic-only mitigation without importing Isaac or CUDA."""
import ast
from pathlib import Path


def test_periodic_stack_walker_stops_before_training_without_disabling_faults():
    source = Path(__file__).resolve().parents[1]/'scripts/train_g1_sonic_sapg.py'
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    names = lambda n: ast.unparse(n.func)
    train = next(n for n in calls if names(n) == 'algo.train')
    enable = next(n for n in calls if names(n) == 'faulthandler.enable')
    timer = next(n for n in calls if names(n) == 'faulthandler.dump_traceback_later')
    cancellation = [n for n in calls if names(n) == 'faulthandler.cancel_dump_traceback_later'
                    and n.lineno < train.lineno]
    assert len(cancellation) == 1
    assert enable.lineno < timer.lineno < cancellation[0].lineno < train.lineno
    assert not any(names(n) == 'faulthandler.disable' for n in calls)
    assert not any(names(n) == 'faulthandler.dump_traceback_later' and n.lineno > train.lineno
                   for n in calls)
    # Cancellation is unconditional in the same try-body as the training call.
    guarded = next(n for n in main.body if isinstance(n, ast.Try) and train in list(ast.walk(n)))
    direct = [n.value for n in guarded.body if isinstance(n, ast.Expr)
              and isinstance(n.value, ast.Call)]
    assert cancellation[0] in direct and train in direct


def test_wandb_thread_starts_before_kit_patches_asyncio():
    source = Path(__file__).resolve().parents[1]/'scripts/train_g1_sonic_sapg.py'
    tree = ast.parse(source.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    initializes = [n for n in calls if ast.unparse(n.func) == 'wandb.init']
    kit = next(n for n in calls if ast.unparse(n.func) == 'AppLauncher')
    assert len(initializes) == 1 and initializes[0].lineno < kit.lineno
