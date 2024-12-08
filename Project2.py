import math
import random
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import csv

neuron_radius = 15
padding = 100

class Neuron:
    def __init__(self, layer_index, neuron_index, x, y):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.x = x
        self.y = y
        self.output = 0.0
        self.delta = 0.0
        self.bias = random.uniform(-1, 1)

    def draw(self, canvas):
        canvas.create_oval(self.x - neuron_radius, self.y - neuron_radius,
                           self.x + neuron_radius, self.y + neuron_radius,
                           fill='white', outline='black')
        canvas.create_text(self.x, self.y, text=f"{self.output:.2f}", font=("Arial", 8))


class Weight:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.value = random.uniform(-1, 1)

    def draw(self, canvas):
        canvas.create_line(self.from_neuron.x, self.from_neuron.y,
                           self.to_neuron.x, self.to_neuron.y, fill='gray')
        mid_x = (self.from_neuron.x + self.to_neuron.x) / 2
        mid_y = (self.from_neuron.y + self.to_neuron.y) / 2
        canvas.create_text(mid_x, mid_y, text=f"{self.value:.2f}", font=("Arial", 8))


class DataPreprocessor:
    @staticmethod
    def load_csv(filepath):
        data = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                try:
                    inputs = [float(x) for x in row[:-1]]
                    target = [float(row[-1])]
                    data.append((inputs, target))
                except ValueError as e:
                    print(f"Skipping invalid row {row}: {e}")
        return data


class NeuralNetwork:
    def __init__(self, layers, ui, activation_function):
        self.layers = layers
        self.neurons = []
        self.weights = []
        self.ui = ui
        self.activation_function = activation_function
        self._build_network()

    def _build_network(self):
        num_layers = len(self.layers)
        layer_width = (self.ui.w - 2 * padding) / (num_layers - 1 if num_layers > 1 else 1)
        self.neurons = []
        for l_index, num_neurons in enumerate(self.layers):
            layer_neurons = []
            layer_height = (self.ui.h - 2 * padding) / (num_neurons - 1 if num_neurons > 1 else 1)
            for n_index in range(num_neurons):
                x = padding + l_index * layer_width
                y = padding + n_index * layer_height
                neuron = Neuron(l_index, n_index, x, y)
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        self.weights = []
        for l in range(len(self.neurons) - 1):
            for from_neuron in self.neurons[l]:
                for to_neuron in self.neurons[l + 1]:
                    weight = Weight(from_neuron, to_neuron)
                    self.weights.append(weight)

    def forward(self, inputs):
        for i, val in enumerate(inputs):
            self.neurons[0][i].output = val

        for l in range(1, len(self.layers)):
            for neuron in self.neurons[l]:
                total_input = neuron.bias
                for prev_neuron in self.neurons[l - 1]:
                    for weight in self.weights:
                        if weight.from_neuron == prev_neuron and weight.to_neuron == neuron:
                            total_input += prev_neuron.output * weight.value
                neuron.output = self._activate(total_input)

        outputs = [neuron.output for neuron in self.neurons[-1]]
        return outputs

    def _activate(self, x):
        if self.activation_function == 'Sigmoid':
            return 1 / (1 + math.exp(-x))
        elif self.activation_function == 'Tanh':
            return math.tanh(x)
        elif self.activation_function == 'ReLU':
            return max(0, x)

    def _activate_derivative(self, output):
        if self.activation_function == 'Sigmoid':
            return output * (1 - output)
        elif self.activation_function == 'Tanh':
            return 1 - (output ** 2)
        elif self.activation_function == 'ReLU':
            return 1 if output > 0 else 0

    def backward(self, targets, lr):
        if len(targets) != len(self.neurons[-1]):
            raise ValueError("Target size does not match output layer size")

        # Output layer deltas
        for i, neuron in enumerate(self.neurons[-1]):
            error = targets[i] - neuron.output
            neuron.delta = error * self._activate_derivative(neuron.output)

        # Hidden layers deltas
        for l in reversed(range(1, len(self.layers) - 1)):
            for neuron in self.neurons[l]:
                error_sum = 0.0
                for weight in self.weights:
                    if weight.from_neuron == neuron:
                        error_sum += weight.to_neuron.delta * weight.value
                neuron.delta = error_sum * self._activate_derivative(neuron.output)

        # Update weights and biases
        for weight in self.weights:
            delta_w = lr * weight.from_neuron.output * weight.to_neuron.delta
            weight.value += delta_w

        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.bias += lr * neuron.delta


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("NN UI")
        self.option_add("*tearOff", FALSE)
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (width, height))
        self.state("zoomed")
        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=width, height=height)
        self.w = width - padding
        self.h = height - padding * 2
        self.network = None
        self.training_data = None
        self.testing_data = None
        self._setup_ui()

    def _setup_ui(self):
        ctrl_frame = Frame(self)
        ctrl_frame.pack(side='left', padx=10, pady=10)

        Label(ctrl_frame, text="Layers (comma-separated):").pack()
        self.layer_entry = Entry(ctrl_frame)
        self.layer_entry.pack()
        self.layer_entry.insert(0, '2,5,7,1')

        Label(ctrl_frame, text="Learning Rate:").pack()
        self.lr_entry = Entry(ctrl_frame)
        self.lr_entry.pack()
        self.lr_entry.insert(0, '0.5')

        Label(ctrl_frame, text="Epochs:").pack()
        self.epoch_entry = Entry(ctrl_frame)
        self.epoch_entry.pack()
        self.epoch_entry.insert(0, '20')

        Label(ctrl_frame, text="Activation Function:").pack()
        self.activation_var = StringVar(value="Sigmoid")
        for func in ['Sigmoid', 'Tanh', 'ReLU']:
            Radiobutton(ctrl_frame, text=func, variable=self.activation_var, value=func).pack(anchor=W)

        Button(ctrl_frame, text="Load CSV", command=self.load_csv).pack(pady=5)
        Button(ctrl_frame, text="Generate Network", command=self.generate_network).pack(pady=5)
        Button(ctrl_frame, text="Start Training", command=self.start_training).pack(pady=5)
        Button(ctrl_frame, text="Start Testing", command=self.start_testing).pack(pady=5)

        self.training_metrics_label = Label(ctrl_frame, text="Training: N/A")
        self.training_metrics_label.pack(pady=5)

        self.testing_metrics_label = Label(ctrl_frame, text="Testing: N/A")
        self.testing_metrics_label.pack(pady=5)

    def generate_network(self):
        layers = [int(s) for s in self.layer_entry.get().split(',')]
        activation_func = self.activation_var.get()
        self.network = NeuralNetwork(layers, self, activation_func)
        self._draw_network()

    def _draw_network(self):
        self.canvas.delete("all")
        for weight in self.network.weights:
            weight.draw(self.canvas)
        for layer in self.network.neurons:
            for neuron in layer:
                neuron.draw(self.canvas)
        self.update()

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            data = DataPreprocessor.load_csv(filepath)
            if len(data) > 0:
                split = int(0.8 * len(data))
                self.training_data = data[:split]
                self.testing_data = data[split:]
                print(f"Loaded {len(self.training_data)} training samples and {len(self.testing_data)} testing samples.")
            else:
                print("No valid data found.")

    def start_training(self):
        if self.network and self.training_data:
            lr = float(self.lr_entry.get())
            epochs = int(self.epoch_entry.get())

            print(f"Training for {epochs} epochs on {len(self.training_data)} samples...")
            for epoch in range(epochs):
                epoch_loss = 0.0
                for i, (inputs, targets) in enumerate(self.training_data, start=1):
                    outputs = self.network.forward(inputs)
                    self.network.backward(targets, lr)
                    loss = sum((t - o)**2 for t, o in zip(targets, outputs)) / len(targets)
                    epoch_loss += loss

                    # Update UI
                    self._draw_network()

                epoch_loss /= len(self.training_data)
                print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")

            # Compute training accuracy and loss
            correct = 0
            final_training_loss = 0.0
            for inputs, targets in self.training_data:
                outputs = self.network.forward(inputs)
                final_training_loss += sum((t - o)**2 for t, o in zip(targets, outputs)) / len(targets)
                if int(round(outputs[0])) == int(targets[0]):
                    correct += 1

            final_training_loss /= len(self.training_data)
            train_acc = (correct / len(self.training_data)) * 100
            print(f"Final Training Accuracy: {train_acc:.2f}% | Final Avg. Training Loss: {final_training_loss:.4f}")
            self.training_metrics_label.config(
                text=f"Training: Acc={train_acc:.2f}% Loss={final_training_loss:.4f}"
            )
            self.update()

    def start_testing(self):
        if self.network and self.testing_data:
            print(f"Testing on {len(self.testing_data)} samples...")
            correct = 0
            total_loss = 0.0
            for i, (inputs, targets) in enumerate(self.testing_data, start=1):
                outputs = self.network.forward(inputs)
                sample_loss = sum((t - o)**2 for t, o in zip(targets, outputs)) / len(targets)
                total_loss += sample_loss

                prediction = int(round(outputs[0]))
                was_correct = (prediction == int(targets[0]))
                correct += int(was_correct)

                # Update UI
                self._draw_network()

            accuracy = (correct / len(self.testing_data)) * 100
            avg_loss = total_loss / len(self.testing_data)
            print(f"Testing complete. Accuracy: {accuracy:.2f}% | Avg. Loss: {avg_loss:.4f}")
            self.testing_metrics_label.config(
                text=f"Testing: Acc={accuracy:.2f}% Loss={avg_loss:.4f}"
            )
            self.update()


if __name__ == '__main__':
    ui = UI()
    ui.mainloop()
