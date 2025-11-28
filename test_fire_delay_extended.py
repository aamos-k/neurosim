#!/usr/bin/env python3
"""Extended test for integrate_and_fire neuron firing delay."""

from neuron_sim import FreeFormNN

def test_with_delay_extended():
    """Test that a neuron with fire_delay=3 fires after 3 steps."""
    print("=" * 60)
    print("EXTENDED TEST: Integrate-and-Fire with 3-step delay")
    print("=" * 60)

    nn = FreeFormNN(["input1"])
    nn.add_neuron("N1", "integrate_and_fire")
    nn.add_connection("N1", "input1", weight=1.0)

    # Set threshold and delay
    nn.neurons["N1"].threshold = 2.0
    nn.neurons["N1"].fire_delay = 3  # 3-step delay

    print(f"Threshold: {nn.neurons['N1'].threshold}")
    print(f"Fire delay: {nn.neurons['N1'].fire_delay}")
    print(f"Input weight: 1.0")
    print()

    # Send constant input of 0.8
    # At step 2: state = 0.8 + 0.8 = 1.6
    # At step 3: state = 1.6 + 0.8 = 2.4 > 2.0, start countdown=3
    # At step 4: countdown=2
    # At step 5: countdown=1
    # At step 6: countdown=0, FIRE! output=1.0, state=0.0
    for step in range(12):
        result = nn.forward([0.8])
        state = nn.neurons["N1"].state
        output = result.get("N1", 0.0)
        countdown = nn.neurons["N1"].fire_countdown
        status = ""
        if countdown >= 0:
            status = f" [COUNTDOWN ACTIVE: {countdown+1} -> {countdown}]"
        if output > 0:
            status = " [FIRED!]"
        print(f"Step {step:02d}: Input=0.8, State={state:.2f}, Countdown={countdown:2d}, Output={output:.1f}{status}")

    print()

def test_no_delay_extended():
    """Test that fire_delay=0 fires immediately."""
    print("=" * 60)
    print("EXTENDED TEST: Integrate-and-Fire with NO delay")
    print("=" * 60)

    nn = FreeFormNN(["input1"])
    nn.add_neuron("N1", "integrate_and_fire")
    nn.add_connection("N1", "input1", weight=1.0)

    # Set threshold, no delay
    nn.neurons["N1"].threshold = 2.0
    nn.neurons["N1"].fire_delay = 0  # No delay

    print(f"Threshold: {nn.neurons['N1'].threshold}")
    print(f"Fire delay: {nn.neurons['N1'].fire_delay}")
    print(f"Input weight: 1.0")
    print()

    # Should fire immediately when threshold is reached
    for step in range(8):
        result = nn.forward([0.8])
        state = nn.neurons["N1"].state
        output = result.get("N1", 0.0)
        countdown = nn.neurons["N1"].fire_countdown
        status = ""
        if output > 0:
            status = " [FIRED IMMEDIATELY!]"
        print(f"Step {step:02d}: Input=0.8, State={state:.2f}, Countdown={countdown:2d}, Output={output:.1f}{status}")

    print()

if __name__ == "__main__":
    test_with_delay_extended()
    test_no_delay_extended()
    print("Extended tests completed!")
