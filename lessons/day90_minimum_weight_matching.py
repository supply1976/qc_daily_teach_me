"""Day 90: Exact minimum-weight matching on a small detection graph."""

from functools import lru_cache

import numpy as np

from day84_spacetime_viterbi_decoder import (
    encoded_state,
    recovery_fidelity,
)
from day89_spacetime_detection_events import (
    detection_matrix,
    edge_weights,
    event_coordinates,
    observed_history,
)


def pair_cost(first, second, spatial_weight, temporal_weight):
    """Weighted Manhattan distance on the repetition-code space-time graph."""
    time_distance = abs(first[0] - second[0])
    check_distance = abs(first[1] - second[1])
    return (
        temporal_weight * time_distance
        + spatial_weight * check_distance
    )


def boundary_cost(event, spatial_weight):
    """Each check is one spatial data edge from its nearest boundary."""
    del event
    return spatial_weight


def minimum_weight_matching(events, spatial_weight, temporal_weight):
    """Exact matching with repeatable spatial-boundary vertices.

    This bitmask dynamic program is exponential and is intended only for
    a tiny teaching example, not a large surface code.
    """
    events = tuple(events)
    count = len(events)

    @lru_cache(maxsize=None)
    def solve(mask):
        if mask == 0:
            return 0.0, ()

        first_index = (mask & -mask).bit_length() - 1
        remaining = mask & ~(1 << first_index)

        best_cost, best_pairs = solve(remaining)
        best_cost += boundary_cost(
            events[first_index],
            spatial_weight,
        )
        best_pairs = (
            ("boundary", first_index, None),
        ) + best_pairs

        candidates = remaining
        while candidates:
            second_index = (
                candidates & -candidates
            ).bit_length() - 1
            next_mask = remaining & ~(1 << second_index)
            rest_cost, rest_pairs = solve(next_mask)
            cost = rest_cost + pair_cost(
                events[first_index],
                events[second_index],
                spatial_weight,
                temporal_weight,
            )
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_pairs = (
                    ("pair", first_index, second_index),
                ) + rest_pairs
            candidates &= ~(1 << second_index)

        return best_cost, best_pairs

    cost, indexed_pairs = solve((1 << count) - 1)
    return cost, materialize_pairs(events, indexed_pairs)


def greedy_matching(events, spatial_weight, temporal_weight):
    """Choose the cheapest immediate action until no events remain."""
    events = tuple(events)
    remaining = set(range(len(events)))
    choices = []
    total_cost = 0.0

    while remaining:
        actions = []
        for index in remaining:
            actions.append((
                boundary_cost(events[index], spatial_weight),
                "boundary",
                index,
                None,
            ))
        ordered = sorted(remaining)
        for position, first_index in enumerate(ordered):
            for second_index in ordered[position + 1:]:
                actions.append((
                    pair_cost(
                        events[first_index],
                        events[second_index],
                        spatial_weight,
                        temporal_weight,
                    ),
                    "pair",
                    first_index,
                    second_index,
                ))

        cost, kind, first_index, second_index = min(actions)
        choices.append((kind, first_index, second_index))
        total_cost += cost
        remaining.remove(first_index)
        if second_index is not None:
            remaining.remove(second_index)

    return total_cost, materialize_pairs(events, tuple(choices))


def materialize_pairs(events, indexed_pairs):
    pairs = []
    for kind, first_index, second_index in indexed_pairs:
        first = events[first_index]
        second = None if second_index is None else events[second_index]
        pairs.append((kind, first, second))
    return tuple(pairs)


def correction_from_matching(matching):
    """Map paths to the parity of inferred final data-X corrections."""
    correction = 0
    for kind, first, second in matching:
        if kind == "boundary":
            # Check 0 borders q0; check 1 borders q2.
            correction ^= 0b001 if first[1] == 0 else 0b100
        elif first[1] != second[1]:
            # A path crossing between the checks crosses middle qubit q1.
            correction ^= 0b010
        # Same-check temporal paths represent readout faults only.
    return correction


def format_matching(matching):
    pieces = []
    for kind, first, second in matching:
        if kind == "boundary":
            pieces.append(f"{first}->B")
        else:
            pieces.append(f"{first}<->{second}")
    return ", ".join(pieces)


def mixed_fault_history(rounds=8):
    """X1 begins at round 3; check-0 readout fails at round 6."""
    observations = observed_history(
        rounds=rounds,
        data_event_round=2,
        data_error_mask=0b010,
        measurement_flips={5: (0,)},
    )
    return observations, 0b010


def find_greedy_counterexample(
    spatial_weight,
    temporal_weight,
    max_round=6,
):
    """Search four-event sets for which greedy matching is suboptimal."""
    vertices = [
        (time, check)
        for time in range(1, max_round + 1)
        for check in range(2)
    ]
    for a in range(len(vertices)):
        for b in range(a + 1, len(vertices)):
            for c in range(b + 1, len(vertices)):
                for d in range(c + 1, len(vertices)):
                    events = (
                        vertices[a],
                        vertices[b],
                        vertices[c],
                        vertices[d],
                    )
                    exact_cost, exact = minimum_weight_matching(
                        events,
                        spatial_weight,
                        temporal_weight,
                    )
                    greedy_cost, greedy = greedy_matching(
                        events,
                        spatial_weight,
                        temporal_weight,
                    )
                    if greedy_cost > exact_cost + 1e-9:
                        return (
                            events,
                            exact_cost,
                            exact,
                            greedy_cost,
                            greedy,
                        )
    return None


def main():
    data_error_probability = 0.02
    readout_error_probability = 0.08
    spatial_weight, temporal_weight = edge_weights(
        data_error_probability,
        readout_error_probability,
    )

    observations, true_final_error = mixed_fault_history()
    detections = detection_matrix(observations)
    events = event_coordinates(detections)
    exact_cost, exact_matching = minimum_weight_matching(
        events,
        spatial_weight,
        temporal_weight,
    )
    inferred_correction = correction_from_matching(exact_matching)

    print("Exact minimum-weight matching on a repetition-code graph")
    print(f"spatial weight: {spatial_weight:.9f}")
    print(f"temporal weight: {temporal_weight:.9f}")
    print("\nMixed-fault syndrome history")
    print("round   syndrome   detection")
    print("-" * 32)
    for time, (syndrome_bits, detection_bits) in enumerate(
        zip(observations, detections),
        start=1,
    ):
        print(
            f"{time:5d}      {syndrome_bits[0]}{syndrome_bits[1]}"
            f"          {detection_bits[0]}{detection_bits[1]}"
        )

    print(f"\nDetection events: {events}")
    print(f"MW matching:      {format_matching(exact_matching)}")
    print(f"total weight:     {exact_cost:.9f}")
    print(f"true final error: {true_final_error:03b}")
    print(f"inferred correction: {inferred_correction:03b}")

    logical_zero = encoded_state(theta=0.0, phase=0.0)
    fidelity = recovery_fidelity(
        logical_zero,
        true_final_error,
        inferred_correction,
    )
    print(f"Qiskit recovery fidelity: {fidelity:.9f}")

    counterexample = find_greedy_counterexample(
        spatial_weight,
        temporal_weight,
    )
    print("\nWhy greedy nearest-neighbor matching is unsafe")
    if counterexample is None:
        print("No counterexample found in the search region.")
    else:
        (
            counter_events,
            counter_exact_cost,
            counter_exact,
            counter_greedy_cost,
            counter_greedy,
        ) = counterexample
        print(f"events:       {counter_events}")
        print(
            f"exact:        {format_matching(counter_exact)}"
            f"  cost={counter_exact_cost:.9f}"
        )
        print(
            f"greedy:       {format_matching(counter_greedy)}"
            f"  cost={counter_greedy_cost:.9f}"
        )
        print(
            "excess weight: "
            f"{counter_greedy_cost - counter_exact_cost:.9f}"
        )

    print("\nComplexity")
    print(f"detection events in example: {len(events)}")
    print(
        "Teaching DP states: at most 2^m = "
        f"{2 ** len(events)} for m events"
    )
    print(
        "Production MWPM uses polynomial-time blossom implementations "
        "instead of this exponential bitmask solver."
    )


if __name__ == "__main__":
    main()
