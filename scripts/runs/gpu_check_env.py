"""Print Python/torch/GPU status for the current conda env (used by gpu_check_envs.sh)."""
import sys

print("python", sys.version.split()[0])
try:
    import torch
    ok = torch.cuda.is_available()
    print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| gpu seen", ok)
    if ok:
        x = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        s = float((x @ x).abs().sum().item())
        print("gpu matmul ok" if s > 0 else "gpu matmul gave zero")
except Exception as e:
    print("torch FAILED:", type(e).__name__, e)
for mod in ("gsplat", "nerfstudio", "pycolmap", "ppisp"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "imports ok"))
    except Exception:
        pass
