# patched_freeform_nn.py
import json
import csv
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict

class Neuron:
    def __init__(self, name, neuron_type="izhikevich"):
        self.name = name
        self.inputs = []  # (source_name, weight, synapse_type)
        self.bias = 0.0
        self.neuron_type = neuron_type.lower()
        self.state = 0.0
        self.output_cache = None

        # Legacy parameters (for backward compatibility)
        self.threshold = 1.0
        self.leak_rate = 0.1

        # Izhikevich model parameters (Regular Spiking by default)
        self.v = -65.0  # Membrane potential (mV)
        self.u = -13.0  # Recovery variable
        self.a = 0.02   # Recovery time scale
        self.b = 0.2    # Sensitivity of u to v
        self.c = -65.0  # After-spike reset value of v
        self.d = 8.0    # After-spike reset increment of u
        self.v_threshold = 30.0  # Spike threshold (mV)

        # --- new fields for advanced dynamics ---
        self.recent_spikes = deque(maxlen=16)  # timestamps (step indices)
        self.burst_remaining = 0
        self.burst_length = 3
        self.burst_interval = 1  # steps between burst pulses (can expand)
        self.stp_state = {}  # src -> current available fraction (1.0..0.0)
        self.stp_recovery = 0.1  # per-step recovery fraction
        self.stp_depression = 0.3  # amount depressed on spike
        self.mod_targets = []  # (target_name, param, scale)
        self.synaptic_fatigue = {}  # src -> fatigue value (0..1)
        self.coincidence_require = 2
        self.coincidence_window = 3  # steps

        # Firing delay mechanism (for integrate_and_fire neuron)
        self.fire_delay = 0  # number of steps to delay before firing
        self.fire_countdown = -1  # countdown state: -1 = not pending, >=0 = steps remaining

    def to_dict(self):
        return {
            "name": self.name,
            "neuron_type": self.neuron_type,
            "inputs": self.inputs,
            "bias": self.bias,
            "threshold": self.threshold,
            "leak_rate": self.leak_rate,
            "state": self.state,
            # Izhikevich parameters
            "v": self.v,
            "u": self.u,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "v_threshold": self.v_threshold,
            # store some new params for reproducibility
            "burst_length": self.burst_length,
            "stp_recovery": self.stp_recovery,
            "stp_depression": self.stp_depression,
            "coincidence_require": self.coincidence_require,
            "coincidence_window": self.coincidence_window,
            # firing delay parameters
            "fire_delay": self.fire_delay,
        }

    @staticmethod
    def from_dict(d):
        n = Neuron(d["name"], d.get("neuron_type", "izhikevich"))
        n.inputs = d.get("inputs", [])
        n.bias = d.get("bias", 0.0)

        # --- Essential Parameters ---
        # Get threshold and leak_rate from the dict, falling back to the class defaults (which are 1.0 and 0.1)
        # However, because Neuron() is called above, we must rely on the dict values or Python's behavior will prioritize the initial __init__ values.
        n.threshold = d.get("threshold", n.threshold)
        n.leak_rate = d.get("leak_rate", n.leak_rate)
        n.state = d.get("state", 0.0)

        # --- Izhikevich Parameters ---
        if "v" in d:
            n.v = d["v"]
        if "u" in d:
            n.u = d["u"]
        if "a" in d:
            n.a = d["a"]
        if "b" in d:
            n.b = d["b"]
        if "c" in d:
            n.c = d["c"]
        if "d" in d:
            n.d = d["d"]
        if "v_threshold" in d:
            n.v_threshold = d["v_threshold"]

        # --- Advanced Parameters (The Source of the Problem) ---
        # The previous code used getattr(n, "param", default) which unnecessarily complicated fetching the default.
        # We simply need to ensure the value from the dict is used, or the initial __init__ value is kept.

        # NOTE: We rely on the initial Neuron.__init__ setting the safe defaults (e.g., burst_length=3).
        # We only override them if they are explicitly present in the loaded dictionary `d`.

        # CORRECTED LOADING: Use the value from the dictionary 'd' if present, otherwise keep the value set in __init__
        if "burst_length" in d:
            n.burst_length = d["burst_length"]

        if "stp_recovery" in d:
            n.stp_recovery = d["stp_recovery"]

        if "stp_depression" in d:
            n.stp_depression = d["stp_depression"]

        if "coincidence_require" in d:
            n.coincidence_require = d["coincidence_require"]

        if "coincidence_window" in d:
            n.coincidence_window = d["coincidence_window"]

        # Firing delay parameters
        if "fire_delay" in d:
            n.fire_delay = d["fire_delay"]

        # This ensures that if the JSON contains "burst_length": 1, it will override the default of 3.
        return n
    def set_bias(self, b):
        self.bias = b

    def add_input(self, source_name, weight=None, input_count=None, synapse_type="S"):
        if weight is None:
            limit = np.sqrt(6 / max(1, input_count))
            weight = np.random.uniform(-limit, limit)
        self.inputs.append((source_name, float(weight), synapse_type))

        # init STP & fatigue bookkeeping
        if source_name not in self.stp_state:
            self.stp_state[source_name] = 1.0
        if source_name not in self.synaptic_fatigue:
            self.synaptic_fatigue[source_name] = 0.0

    def clear_cache(self):
        self.output_cache = None

    def get_spike_proximity(self):
        """Returns a value 0-1 indicating how close the neuron is to spiking.
        1.0 means very close (near threshold), 0.0 means far from spiking."""
        if self.neuron_type == "izhikevich":
            # Map voltage range to 0-1
            # Typical range: v_rest (-65) to v_threshold (30)
            v_rest = -65.0
            v_range = self.v_threshold - v_rest  # 95 mV
            normalized = (self.v - v_rest) / v_range
            return max(0.0, min(1.0, normalized))
        elif self.neuron_type in ["leaky_integrate_and_fire", "integrate_and_fire"]:
            normalized = self.state / self.threshold
            return max(0.0, min(1.0, normalized))
        else:
            return 0.0

    def activate(self, x, step_index=None):
        if self.neuron_type == "izhikevich":
            # Izhikevich neuron model
            # Input current (scaled to mV range)
            I = (x + self.bias) * 10.0  # Scale input to appropriate range

            # Integration timestep (ms)
            dt = 0.5

            # Euler integration with sub-steps for stability
            for _ in range(2):
                # dv/dt = 0.04v² + 5v + 140 - u + I
                dv = (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I) * dt
                # du/dt = a(bv - u)
                du = self.a * (self.b * self.v - self.u) * dt

                self.v += dv
                self.u += du

            # Check for spike
            if self.v >= self.v_threshold:
                self.v = self.c  # Reset voltage
                self.u += self.d  # Reset recovery variable
                if step_index is not None:
                    self.recent_spikes.append(step_index)
                return 1.0
            else:
                return 0.0

        elif self.neuron_type == "integrate_and_fire":
            # Check if we're in countdown mode
            if self.fire_countdown >= 0:
                self.fire_countdown -= 1
                if self.fire_countdown == 0:
                    # Fire after delay
                    self.fire_countdown = -1
                    self.state = 0.0
                    if step_index is not None:
                        self.recent_spikes.append(step_index)
                    return 1.0
                else:
                    # Still counting down, don't fire yet
                    return 0.0

            # Normal integration
            self.state += x + self.bias
            if self.state >= self.threshold:
                if self.fire_delay > 0:
                    # Start countdown instead of firing immediately
                    self.fire_countdown = self.fire_delay
                    return 0.0
                else:
                    # Fire immediately (backward compatible behavior)
                    self.state = 0.0
                    if step_index is not None:
                        self.recent_spikes.append(step_index)
                    return 1.0
            else:
                return 0.0
        elif self.neuron_type == "leaky_integrate_and_fire":
            self.state = (1 - self.leak_rate) * self.state + x + self.bias
            if self.state >= self.threshold:
                self.state = -0.5 * self.threshold
                if step_index is not None:
                    self.recent_spikes.append(step_index)
                return 1.0
            else:
                return 0.0
        else:
            # Default to Izhikevich model if unknown type
            I = (x + self.bias) * 10.0
            dt = 0.5
            for _ in range(2):
                dv = (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I) * dt
                du = self.a * (self.b * self.v - self.u) * dt
                self.v += dv
                self.u += du

            if self.v >= self.v_threshold:
                self.v = self.c
                self.u += self.d
                if step_index is not None:
                    self.recent_spikes.append(step_index)
                return 1.0
            else:
                return 0.0

    def compute_output(self, values, step_index=0):
        # caching per-step handled externally by clear_all_cache
        if self.output_cache is not None:
            return self.output_cache

        total_input = 0.0
        # sum inputs, applying STP depression and per-synapse fatigue
        for src, w, syn_type in self.inputs:
            src_val = values.get(src, 0.0)
            eff_w = float(w)

            # STP synapse type handling
            if syn_type == "STP":
                avail = self.stp_state.get(src, 1.0)
                eff_w *= avail
            # Synaptic fatigue (generic)
            fatigue = self.synaptic_fatigue.get(src, 0.0)
            eff_w *= (1.0 - fatigue)

            total_input += src_val * eff_w

        output = self.activate(total_input, step_index=step_index)

        # If any STP synapses have a presynaptic spike this step, depress them
        for src, _, syn_type in self.inputs:
            if syn_type == "STP":
                # If presynaptic value present and >0, assume it was a spike this step
                if values.get(src, 0.0) > 0:
                    self.stp_state[src] = max(0.0, self.stp_state.get(src, 1.0) - self.stp_depression)

        # Recover STP states a little
        for src in list(self.stp_state.keys()):
            self.stp_state[src] = min(1.0, self.stp_state[src] + self.stp_recovery)

        # Simple fatigue increase if neuron fired this step
        if output > 0:
            for src, _, _ in self.inputs:
                # increase fatigue on those synapses a little
                self.synaptic_fatigue[src] = min(1.0, self.synaptic_fatigue.get(src, 0.0) + 0.02)
        else:
            # recover fatigue
            for src in list(self.synaptic_fatigue.keys()):
                self.synaptic_fatigue[src] = max(0.0, self.synaptic_fatigue[src] - 0.01)

        self.output_cache = float(output)
        return self.output_cache

    def hebbian_update(self, input_values, output, learning_rate=0.01):
        new_inputs = []
        for src, w, syn_type in self.inputs:
            inp_val = input_values.get(src, 0)
            dw = learning_rate * output * inp_val
            new_inputs.append((src, w + dw, syn_type))
        self.inputs = new_inputs
        self.bias += learning_rate * output

class FreeFormNN:
    def __init__(self, input_names):
        self.input_names = list(input_names)
        self.neurons = {}
        self.step_index = 0  # global step counter for recent_spikes timestamps

    def add_neuron(self, name, neuron_type="sigmoid"):
        if name in self.neurons or name in self.input_names:
            raise ValueError(f"Neuron name '{name}' conflicts with inputs or existing neuron.")
        self.neurons[name] = Neuron(name, neuron_type)

    def set_bias(self, neuron_name, bias):
        self.neurons[neuron_name].set_bias(bias)

    def add_connection(self, neuron_name, source_name, weight=None, synapse_type="S"):
        if neuron_name not in self.neurons:
            raise ValueError(f"Neuron '{neuron_name}' does not exist.")
        if source_name not in self.input_names and source_name not in self.neurons:
            raise ValueError(f"Source '{source_name}' not found.")
        input_count = len(self.neurons[neuron_name].inputs) + 1
        self.neurons[neuron_name].add_input(source_name, weight, input_count, synapse_type)

        if synapse_type == "EJ" and source_name in self.neurons:
            reverse_input_count = len(self.neurons[source_name].inputs) + 1
            # add reciprocal weak connection automatically
            self.neurons[source_name].add_input(neuron_name, weight, reverse_input_count, "EJ")

    def clear_all_cache(self):
        for neuron in self.neurons.values():
            neuron.clear_cache()
            
    def forward(self, input_values, steps=1):
        if len(input_values) != len(self.input_names):
            raise ValueError("Input length mismatch.")
        values = dict(zip(self.input_names, input_values))
        history = []

        for _ in range(steps):
            self.clear_all_cache()
            current_outputs = {}

            # Compute neuron outputs in insertion order, immediately updating 'values'
            for neuron in self.neurons.values():
                out = neuron.compute_output(values, step_index=self.step_index)
                current_outputs[neuron.name] = out
                # Make output immediately available for downstream neurons
                values[neuron.name] = out

            history.append(dict(values))
            self.step_index += 1

        return history[-1] if history else {}


    def train_hebbian(self, input_values, learning_rate=0.01):
        outputs = self.forward(input_values)
        values = dict(zip(self.input_names, input_values))
        values.update(outputs)
        for neuron in self.neurons.values():
            neuron.hebbian_update(values, outputs[neuron.name], learning_rate)

    def save(self, filename):
        data = {
            "input_names": self.input_names,
            "neurons": [n.to_dict() for n in self.neurons.values()]
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Network saved to '{filename}'")

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        net = FreeFormNN(data["input_names"])
        for nd in data["neurons"]:
            net.neurons[nd["name"]] = Neuron.from_dict(nd)
        print(f"Network loaded from '{filename}'")
        return net

    def visualize(self, activations=None):
        """Display network connections in terminal format."""
        print("\n" + "="*80)
        print("NEURAL NETWORK STRUCTURE")
        print("="*80)

        # Display input neurons
        print("\nINPUT NEURONS:")
        for inp in self.input_names:
            print(f"  {inp}")

        # Display all neurons and their connections
        print("\nNEURONS AND CONNECTIONS:")
        for neuron in self.neurons.values():
            bias = neuron.bias
            neuron_display = f"{neuron.name} (bias: {bias:.2f})"
            print(f"\n  {neuron_display}:")

            if not neuron.inputs:
                print("    [No incoming connections]")
            else:
                for src, weight, syn_type in neuron.inputs:
                    # Get source bias if it's a neuron
                    if src in self.neurons:
                        src_bias = self.neurons[src].bias
                        src_display = f"{src} (bias: {src_bias:.2f})"
                    else:
                        src_display = src

                    print(f"    {src_display} → {weight:.2f} → {neuron.name}")

        print("\n" + "="*80 + "\n")



def load_csv_network(filename, input_ids=None):
    input_ids = input_ids or []
    nn = FreeFormNN(input_ids)
    neuron_ids = set()

    # Try auto-detecting delimiter
    with open(filename, newline='') as f:
        sample = f.read(1024)
        delimiter = '\t' if '\t' in sample else ','
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)
        expected_fields = {'pre_root_id', 'post_root_id', 'syn_count', 'nt_type'}
        if not expected_fields.issubset(reader.fieldnames):
            raise ValueError(f"CSV format error: Expected headers not found. Got: {reader.fieldnames}")

        for row in reader:
            try:
                pre = str(int(float(row['pre_root_id'])))
                post = str(int(float(row['post_root_id'])))
                weight = float(row['syn_count'])
                nt = row['nt_type'].strip().upper()
            except KeyError as e:
                raise ValueError(f"Missing column in CSV: {e}")

            syn_type = "S"
            if nt == "GABA":
                syn_type = "R"
                weight *= -1
            elif nt == "GLUT":
                syn_type = "S"
            elif nt == "ACH":
                syn_type = "Sp"

            for n in [pre, post]:
                if n not in neuron_ids and n not in input_ids:
                    nn.add_neuron(n, "izhikevich")
                    neuron_ids.add(n)

            nn.add_connection(post, pre, weight=weight, synapse_type=syn_type)
    if not input_ids:
        out_counts = {}  # source -> number of outgoing connections
        in_counts = {}   # target -> number of incoming connections
        for neuron in nn.neurons.values():
            for src, _, _ in neuron.inputs:
                in_counts[neuron.name] = in_counts.get(neuron.name, 0) + 1
                out_counts[src] = out_counts.get(src, 0) + 1
    
        total_neurons = len(nn.neurons)
        n_inputs = max(1, total_neurons // 100)   # ~5% as input
        n_outputs = max(1, total_neurons // 100)  # ~5% as output
    
        top_inputs = sorted(out_counts, key=out_counts.get, reverse=True)[:n_inputs]
        top_outputs = sorted(in_counts, key=in_counts.get, reverse=True)[-n_outputs:]
    
        # Deduplicate
        top_inputs = [nid for nid in top_inputs if nid not in nn.input_names]
        top_outputs = [nid for nid in top_outputs if nid not in top_inputs]
    
        nn.input_names.extend(top_inputs)
        print(f"Auto-selected {len(top_inputs)} input neurons: {top_inputs}")
        print(f"Auto-selected {len(top_outputs)} output neurons: {top_outputs}")

    print(f"Loaded network from '{filename}' with {len(neuron_ids)} neurons and input(s): {input_ids}")
    return nn
    
def generate_random_LIF_network(n_inputs=5, n_neurons=50, n_outputs=5, connection_prob=0.1):
    # Create the network
    nn = FreeFormNN([f"in{i}" for i in range(n_inputs)])
    
    # Add hidden neurons
    for i in range(n_neurons):
        nn.add_neuron(f"h{i}", neuron_type="izhikevich")

    # Add output neurons
    for i in range(n_outputs):
        nn.add_neuron(f"out{i}", neuron_type="izhikevich")
    
    all_neurons = list(nn.neurons.keys())
    
    # Randomly connect neurons
    for target in all_neurons:
        # skip inputs
        if target in nn.input_names:
            continue
        for source in nn.input_names + [n for n in all_neurons if n != target]:
            if np.random.rand() < connection_prob:
                weight = np.random.uniform(-1.0, 1.0)
                nn.add_connection(target, source, weight=weight, synapse_type="S")
    
    # Optional: pick top outputs (neurons with few outgoing connections)
    out_candidates = sorted(all_neurons, key=lambda n: len(nn.neurons[n].inputs))
    nn.input_names = nn.input_names  # keep inputs
    print(f"Generated network with {len(nn.neurons)} neurons, inputs={nn.input_names}")
    return nn


def load_xls_network(filename, input_names=None):

    input_names = set(input_names or [])
    nn = FreeFormNN([])  # Set input_names later after detecting

    neuron_ids = set()
    r_type_inputs = set()

    df = pd.read_excel(filename, engine="openpyxl" if filename.endswith("xlsx") else "xlrd")

    required_cols = ["Neuron 1", "Neuron 2", "Type", "Nbr"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one of the required columns: {required_cols}. Found: {df.columns.tolist()}")

    for _, row in df.iterrows():
        pre = str(row["Neuron 1"]).strip()
        post = str(row["Neuron 2"]).strip()
        syn_type = str(row["Type"]).strip().upper()
        count = float(row["Nbr"])

        # Track neurons that should become inputs
        if syn_type in ("R", "RP"):
            r_type_inputs.add(post)

        # Create neurons if not present
        for n in (pre, post):
            if n not in neuron_ids and n not in input_names:
                nn.add_neuron(n, "izhikevich")
                neuron_ids.add(n)

        nn.add_connection(post, pre, weight=count, synapse_type=syn_type)

    # Add R-type receivers to input list
    input_names.update(r_type_inputs)
    nn.input_names = list(input_names)

    print(f"Auto-assigned {len(r_type_inputs)} input neurons from R/Rp connections.")
    print(f"Total neurons: {len(nn.neurons)}")

    return nn


def calculate_adaptive_steps(nn, min_skip=1, max_skip=20):
    """Calculate how many steps to skip between outputs based on spike proximity.
    When neurons are close to spiking, skip fewer steps (more frequent output).
    When far from spiking, skip more steps (less frequent output).
    Uses aggressive exponential scaling to concentrate outputs near spikes."""
    if nn is None or not nn.neurons:
        return max_skip

    max_proximity = 0.0
    for neuron in nn.neurons.values():
        proximity = neuron.get_spike_proximity()
        max_proximity = max(max_proximity, proximity)

    # Inverse relationship: high proximity -> low skip count (frequent output)
    # Use aggressive exponential scaling (power of 4) for dramatic effect near spikes
    # This makes the system output rarely when far from spikes,
    # then output very frequently when approaching threshold
    skip_range = max_skip - min_skip
    steps_to_skip = max_skip - skip_range * (max_proximity ** 4)
    return max(min_skip, int(steps_to_skip))

def main():
    print("Free-form Neural Network Shell")
    print("Supported neuron types: izhikevich, integrate_and_fire, leaky_integrate_and_fire")
    nn = None

    while True:
        cmd = input("\nCommands:\n"
                    " create [inputs...]\n"
                    " add_neuron [name] [type: izhikevich|integrate_and_fire|leaky_integrate_and_fire]\n"
                    " set_bias [neuron] [bias]\n"
                    " add_conn [target] [source] [weight (opt)] [type (S, Sp, R, Rp, EJ, NMJ)]\n"
                    " run_step [steps] [min] [max]\n"
                    " run_pulse [pulses]\n"
                    " train [inputs...] [learning_rate (opt)]\n"
                    " save [filename]\n"
                    " load [filename]\n"
                    " load_csv [csv_file] [optional: input ids...]\n"
                    " load_xls [filename] [optional input1 input2 ...]\n"
                    " visualize\n"
                    " add_input [input_name]\n"
                    " pulse_input [input_name] [voltage] [steps_on] [steps_off]\n"
                    " quit\n> ")
        parts = cmd.strip().split()
        if not parts:
            continue
        action = parts[0].lower()

        try:
            if action == "quit":
                break
            elif action == "create":
                nn = FreeFormNN(parts[1:])
                print(f"Created network with inputs: {nn.input_names}")
            elif action == "add_neuron":
                if nn is None: print("Create network first"); continue
                name = parts[1]
                ntype = parts[2] if len(parts) > 2 else "izhikevich"
                nn.add_neuron(name, ntype)
                print(f"Added neuron '{name}' of type '{ntype}'")
            elif action == "set_bias":
                if nn is None: print("Create network first"); continue
                nn.set_bias(parts[1], float(parts[2]))
            elif action == "add_conn":
                if nn is None: print("Create network first"); continue
                neuron = parts[1]
                source = parts[2]
                weight = float(parts[3]) if len(parts) > 3 and parts[3] not in ["S", "Sp", "R", "Rp", "EJ", "NMJ"] else None
                syn_type = parts[4] if len(parts) > 4 else ("S" if weight is not None else parts[3] if len(parts) > 3 else "S")
                nn.add_connection(neuron, source, weight, syn_type)
                print(f"Connection: {source} → {neuron}, weight={weight if weight else '(auto)'}, type={syn_type}")
            elif action == "run_step":
                if nn is None: print("Create network first"); continue
                total_steps = int(parts[1]) if len(parts) > 1 else 1000
                vmin = float(parts[2]) if len(parts) > 2 else 0.0
                vmax = float(parts[3]) if len(parts) > 3 else 1.0

                inputs = [0.0 for _ in nn.input_names]
                history = []

                print("Adaptive output: displays more frequently near spikes")
                print("-" * 80)

                sim_step = 0
                output_count = 0
                while sim_step < total_steps:
                    # Random walk input
                    for i in range(len(inputs)):
                        change = np.random.uniform(-0.1, 0.1)
                        inputs[i] = max(vmin, min(vmax, inputs[i] + change))

                    output = nn.forward(inputs)
                    history.append((list(inputs), dict(output)))

                    # Calculate how many steps to skip before next output
                    skip_steps = calculate_adaptive_steps(nn)
                    max_prox = max([n.get_spike_proximity() for n in nn.neurons.values()] or [0.0])

                    print(f"Step {sim_step:04d} | Input: {['%.2f' % x for x in inputs]} | Output: " +
                          ', '.join(f"{k}:{v:.2f}" for k, v in output.items()) +
                          f" | Proximity: {max_prox:.2f} | Skip: {skip_steps}")

                    # Advance simulation by skip_steps (run hidden steps)
                    for _ in range(skip_steps - 1):
                        for i in range(len(inputs)):
                            change = np.random.uniform(-0.1, 0.1)
                            inputs[i] = max(vmin, min(vmax, inputs[i] + change))
                        nn.forward(inputs)
                        sim_step += 1
                        if sim_step >= total_steps:
                            break

                    sim_step += 1
                    output_count += 1
                    time.sleep(0.01)  # Minimal delay for readability

                print("-" * 80)
                print(f"Total simulation steps: {sim_step}, Outputs shown: {output_count}")

            elif action == "run_pulse":
                if nn is None: print("Create network first"); continue
                total_steps = int(parts[1]) if len(parts) > 1 else 1000
                history = []

                print("Adaptive output: displays more frequently near spikes")
                print("-" * 80)

                sim_step = 0
                output_count = 0
                while sim_step < total_steps:
                    inputs = [np.random.choice([0.0, 1.0]) for _ in nn.input_names]
                    output = nn.forward(inputs)
                    history.append((list(inputs), dict(output)))

                    # Calculate how many steps to skip before next output
                    skip_steps = calculate_adaptive_steps(nn)
                    max_prox = max([n.get_spike_proximity() for n in nn.neurons.values()] or [0.0])

                    print(f"Step {sim_step:04d} | Input: {inputs} | Output: " +
                          ', '.join(f"{k}:{v:.2f}" for k, v in output.items()) +
                          f" | Proximity: {max_prox:.2f} | Skip: {skip_steps}")

                    # Advance simulation by skip_steps (run hidden steps)
                    for _ in range(skip_steps - 1):
                        inputs = [np.random.choice([0.0, 1.0]) for _ in nn.input_names]
                        nn.forward(inputs)
                        sim_step += 1
                        if sim_step >= total_steps:
                            break

                    sim_step += 1
                    output_count += 1
                    time.sleep(0.01)  # Minimal delay for readability

                print("-" * 80)
                print(f"Total simulation steps: {sim_step}, Outputs shown: {output_count}")
            
            elif action == "train":
                if nn is None: print("Create network first"); continue
                learning_rate = 0.01
                if len(parts) > len(nn.input_names) + 1:
                    learning_rate = float(parts[-1])
                    inputs = list(map(float, parts[1:-1]))
                else:
                    inputs = list(map(float, parts[1:]))
                nn.train_hebbian(inputs, learning_rate)
                print("Hebbian training step done.")
            elif action == "save":
                nn.save(parts[1])
            elif action == "load":
                nn = FreeFormNN.load(parts[1])
            elif action == "load_csv":
                filename = parts[1]
                input_ids = parts[2:]
                nn = load_csv_network(filename, input_ids)
            elif action == "load_xls":
                filename = parts[1]
                user_inputs = parts[2:] if len(parts) > 2 else []
                nn = load_xls_network(filename, user_inputs)
            elif action == "visualize":
                if nn is None:
                    print("Create or load a network first.")
                    continue
                print("Visualizing current network...")
                nn.visualize()
            elif action == "add_input":
                if nn is None:
                    print("Create network first")
                    continue
                if len(parts) < 2:
                    print("Usage: add_input [input_name]")
                    continue
                input_name = parts[1].strip()
                if input_name in nn.input_names:
                    print(f"Input '{input_name}' already exists.")
                else:
                    nn.input_names.append(input_name)
                    print(f"Added new input '{input_name}'.")
            # Replace the pulse_input command section in main() with this:
 
            elif action == "pulse_input":
                if nn is None:
                    print("Create a network first")
                    continue
            
                if len(parts) < 5:
                    print("Usage: pulse_input [input_name] [voltage] [steps_on] [steps_off]")
                    continue
            
                input_name = parts[1]
                if input_name not in nn.input_names:
                    print(f"Input '{input_name}' is not defined.")
                    continue
            
                voltage = float(parts[2])
                steps_on = int(parts[3])
                steps_off = int(parts[4])
                
                # Get the index of the target input
                input_idx = nn.input_names.index(input_name)
                
                # Initialize baseline inputs (all zeros)
                inputs = [0.0 for _ in nn.input_names]
                
                print(f"\nPulsing input '{input_name}' with {voltage}V for {steps_on} steps ON, {steps_off} steps OFF")
                print("Adaptive output: displays more frequently near spikes")
                print("-" * 80)

                # Simulate with voltage ON
                sim_step = 0
                output_count = 0
                while sim_step < steps_on:
                    inputs[input_idx] = voltage
                    output = nn.forward(inputs)

                    # Calculate how many steps to skip before next output
                    skip_steps = calculate_adaptive_steps(nn)
                    max_prox = max([n.get_spike_proximity() for n in nn.neurons.values()] or [0.0])

                    print(f"Step {sim_step:04d} (ON)  | Input: {['%.2f' % x for x in inputs]} | Output: " +
                          ', '.join(f"{k}:{v:.2f}" for k, v in output.items()) +
                          f" | Proximity: {max_prox:.2f} | Skip: {skip_steps}")

                    # Advance simulation by skip_steps (run hidden steps)
                    for _ in range(skip_steps - 1):
                        inputs[input_idx] = voltage
                        nn.forward(inputs)
                        sim_step += 1
                        if sim_step >= steps_on:
                            break

                    sim_step += 1
                    output_count += 1
                    time.sleep(0.01)  # Minimal delay for readability

                # Simulate with voltage OFF
                sim_step = 0
                while sim_step < steps_off:
                    inputs[input_idx] = 0.0
                    output = nn.forward(inputs)

                    # Calculate how many steps to skip before next output
                    skip_steps = calculate_adaptive_steps(nn)
                    max_prox = max([n.get_spike_proximity() for n in nn.neurons.values()] or [0.0])

                    print(f"Step {sim_step:04d} (OFF) | Input: {['%.2f' % x for x in inputs]} | Output: " +
                          ', '.join(f"{k}:{v:.2f}" for k, v in output.items()) +
                          f" | Proximity: {max_prox:.2f} | Skip: {skip_steps}")

                    # Advance simulation by skip_steps (run hidden steps)
                    for _ in range(skip_steps - 1):
                        inputs[input_idx] = 0.0
                        nn.forward(inputs)
                        sim_step += 1
                        if sim_step >= steps_off:
                            break

                    sim_step += 1
                    output_count += 1
                    time.sleep(0.01)  # Minimal delay for readability

                print("-" * 80)
                print(f"Pulse sequence complete. Total outputs shown: {output_count}\n")
            elif action == "generate":
                if nn is None:
                    print("Create network first (use 'create [inputs...]').")
                    continue
                try:
                    n_inputs = len(nn.input_names)
                    n_hidden = int(parts[1]) if len(parts) > 1 else 5
                    n_outputs = int(parts[2]) if len(parts) > 2 else 2
            
                    hidden_names = [f"H{i}" for i in range(n_hidden)]
                    output_names = [f"O{i}" for i in range(n_outputs)]
            
                    # Add hidden neurons
                    for hn in hidden_names:
                        nn.add_neuron(hn, "izhikevich")

                    # Add output neurons
                    for on in output_names:
                        nn.add_neuron(on, "izhikevich")
            
                    # Connect inputs to hidden
                    for hn in hidden_names:
                        for inp in nn.input_names:
                            nn.add_connection(hn, inp)
            
                    # Connect hidden to outputs
                    for on in output_names:
                        for hn in hidden_names:
                            nn.add_connection(on, hn)
            
                    print(f"Generated network with {n_hidden} hidden neurons and {n_outputs} outputs.")
                except Exception as e:
                    print(f"Error generating network: {e}")
            
            else:
                print("Unknown command.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()


