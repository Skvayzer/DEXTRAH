"""Deterministic selection of a continuous recording, not a success benchmark."""


def summarize_candidate(steps, policy_dt, lift_threshold):
    longest = run = 0
    first_lift = first_goal = None
    best_end = None
    for metric in steps:
        if metric["lift_height_m"] > lift_threshold:
            run += 1
            if first_lift is None:
                first_lift = metric["step"] * policy_dt
            if run > longest:
                longest, best_end = run, metric["step"]
        else:
            run = 0
        if metric["goal_hit"] and first_goal is None:
            first_goal = metric["step"] * policy_dt
        # Do not join two episodes into one apparent sustained lift.
        if metric["done"]:
            run = 0
    return dict(
        goal_hits=sum(int(s["goal_hit"]) for s in steps),
        resets=sum(int(s["done"]) for s in steps),
        total_reward=sum(s["reward"] for s in steps),
        peak_lift_height_m=max(s["lift_height_m"] for s in steps),
        longest_lift_seconds=longest * policy_dt,
        longest_lift_interval_s=(None if best_end is None else
                                [(best_end - longest) * policy_dt, best_end * policy_dt]),
        first_lift_seconds=first_lift, first_goal_seconds=first_goal,
    )


def select_candidate(summaries):
    if not summaries:
        raise ValueError("No recorded candidates")
    # Never present reward alone as evidence of task success.
    return max(range(len(summaries)), key=lambda i: (
        summaries[i]["goal_hits"], summaries[i]["longest_lift_seconds"],
        summaries[i]["total_reward"], -i,
    ))
