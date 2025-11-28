#!/usr/bin/env python3
"""Test script for integrate_and_fire neuron firing delay."""

from neuron_sim import FreeFormNN

def test_no_delay():
    """Test that a neuron with fire_delay=0 fires immediately."""
    print("=" * 60)
    print("TEST 1: Integrate-and-Fire with NO delay (fire_delay=0)")
    print("=" * 60)

    nn = FreeFormNN(["input1"])
    nn.add_neuron("N1", "integrate_and_fire")
    nn.add_connection("N1", "input1", weight=0.4)

    # Set threshold to 1.0 (default)
    nn.neurons["N1"].threshold = 1.0
    nn.neurons["N1"].fire_delay = 0  # No delay

    print(f"Threshold: {nn.neurons['N1'].threshold}")
    print(f"Fire delay: {nn.neurons['N1'].fire_delay}")
    print(f"Input weight: 0.4")
    print()

    # Send input that will reach threshold in 3 steps
    # 0.4 + 0.4 + 0.4 = 1.2 > 1.0
    for step in range(5):
        result = nn.forward([0.4])
        state = nn.neurons["N1"].state
        output = result.get("N1", 0.0)
        countdown = nn.neurons["N1"].fire_countdown
        print(f"Step {step}: State={state:.2f}, Countdown={countdown}, Output={output:.1f}")

    print()

def test_with_delay():
    """Test that a neuron with fire_delay=3 fires after 3 steps."""
    print("=" * 60)
    print("TEST 2: Integrate-and-Fire with 3-step delay (fire_delay=3)")
    print("=" * 60)

    nn = FreeFormNN(["input1"])
    nn.add_neuron("N1", "integrate_and_fire")
    nn.add_connection("N1", "input1", weight=0.4)

    # Set threshold and delay
    nn.neurons["N1"].threshold = 1.0
    nn.neurons["N1"].fire_delay = 3  # 3-step delay

    print(f"Threshold: {nn.neurons['N1'].threshold}")
    print(f"Fire delay: {nn.neurons['N1'].fire_delay}")
    print(f"Input weight: 0.4")
    print()

    # Send input that will reach threshold in 3 steps
    for step in range(8):
        result = nn.forward([0.4])
        state = nn.neurons["N1"].state
        output = result.get("N1", 0.0)
        countdown = nn.neurons["N1"].fire_countdown
        print(f"Step {step}: State={state:.2f}, Countdown={countdown}, Output={output:.1f}")

    print()

def test_persistence():
    """Test that fire_delay is saved and loaded correctly."""
    print("=" * 60)
    print("TEST 3: Save and Load fire_delay parameter")
    print("=" * 60)

    # Create network with delay
    nn = FreeFormNN(["input1"])
    nn.add_neuron("N1", "integrate_and_fire")
    nn.neurons["N1"].fire_delay = 5

    # Save to file
    nn.save("/tmp/test_delay_network.json")
    print(f"Created neuron with fire_delay={nn.neurons['N1'].fire_delay}")

    # Load from file
    nn2 = FreeFormNN.load("/tmp/test_delay_network.json")
    print(f"Loaded neuron with fire_delay={nn2.neurons['N1'].fire_delay}")

    if nn2.neurons['N1'].fire_delay == 5:
        print("✓ Persistence test passed!")
    else:
        print("✗ Persistence test failed!")

    print()

if __name__ == "__main__":
    test_no_delay()
    test_with_delay()
    test_persistence()
    print("All tests completed!")
