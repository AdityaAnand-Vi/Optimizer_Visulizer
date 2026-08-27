import json
from main import simulate_2d, Simulate2DRequest, train_nn, TrainNNRequest

results = {}

# EXPERIMENT 1: Learning Rate
req1_1 = Simulate2DRequest(surface_name="L2: c = 50 (default)", optimizers=["SGD"], x0=8.0, y0=8.0, lr=0.001, max_steps=500)
res1_1 = simulate_2d(req1_1)
req1_2 = Simulate2DRequest(surface_name="L2: c = 50 (default)", optimizers=["SGD"], x0=8.0, y0=8.0, lr=0.01, max_steps=500)
res1_2 = simulate_2d(req1_2)
req1_3 = Simulate2DRequest(surface_name="L2: c = 50 (default)", optimizers=["SGD"], x0=8.0, y0=8.0, lr=0.1, max_steps=500)
res1_3 = simulate_2d(req1_3)

results["EXP1"] = {
    "0.001": {"loss": res1_1["losses"]["SGD"][-1], "pos": res1_1["trajectories"]["SGD"][-1], "status": res1_1["final_statuses"]["SGD"]},
    "0.01": {"loss": res1_2["losses"]["SGD"][-1], "pos": res1_2["trajectories"]["SGD"][-1], "status": res1_2["final_statuses"]["SGD"]},
    "0.1": {"loss": res1_3["losses"]["SGD"][-1], "pos": res1_3["trajectories"]["SGD"][-1], "status": res1_3["final_statuses"]["SGD"]},
}

# EXPERIMENT 2: Optimizer Comparison
req2 = Simulate2DRequest(surface_name="L2: c = 50 (default)", optimizers=["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"], x0=8.0, y0=8.0, lr=0.01, max_steps=500)
res2 = simulate_2d(req2)
results["EXP2"] = {}
for opt in req2.optimizers:
    results["EXP2"][opt] = {
        "loss": res2["losses"][opt][-1],
        "pos": res2["trajectories"][opt][-1],
        "status": res2["final_statuses"][opt],
        "steps": len(res2["losses"][opt]) - 1
    }

# EXPERIMENT 5: Neural Benchmark
req5 = TrainNNRequest(optimizers=["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"], lr=0.01, epochs=100, batch_size=32)
res5 = train_nn(req5)
results["EXP5"] = {}
for opt in req5.optimizers:
    summary = next(s for s in res5["summary_table"] if s["optimizer"] == opt)
    results["EXP5"][opt] = summary

# EXPERIMENT 6: Divergence
req6 = Simulate2DRequest(surface_name="L3: c = 100", optimizers=["SGD"], x0=8.0, y0=8.0, lr=0.1, max_steps=500)
res6 = simulate_2d(req6)
results["EXP6"] = {
    "loss": res6["losses"]["SGD"][-1],
    "pos": res6["trajectories"]["SGD"][-1],
    "status": res6["final_statuses"]["SGD"]
}

with open("empirical_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done generating empirical results.")
