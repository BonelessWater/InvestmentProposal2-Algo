from manim import *
import numpy as np

class NeuralNetworkVisualization(Scene):
    def construct(self):
        # Set the background color to white.
        self.camera.background_color = WHITE

        # Introduction title with black text.
        title = Text("Understanding Neural Networks", font_size=42, color=BLACK)
        subtitle = Text("From Architecture to Activation Functions", font_size=24, color=BLACK)
        subtitle.next_to(title, DOWN, buff=0.5)
        intro_group = VGroup(title, subtitle)
        intro_group.center()
        self.play(Write(intro_group))
        self.wait(1.5)
        self.play(FadeOut(intro_group))

        # Create a neural network (scaled-down version)
        network = self.create_neural_network([4, 6, 5, 3], node_radius=0.2)
        self.play(Create(network))
        self.wait(2)

        # Highlight the flow of information.
        self.highlight_network_flow(network)
        self.wait(1)

        self.play(FadeOut(network))
        self.wait(1)

        # Demonstrate forward pass with two inputs activating one neuron.
        self.demonstrate_two_inputs_activation()
        self.wait(2)

        # Now demonstrate backpropagation.
        self.demonstrate_backpropagation()
        self.wait(2)

    def demonstrate_two_inputs_activation(self):
        # Create two input nodes with activation values.
        input_node1 = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        input_node1.move_to(LEFT * 3 + UP * 1)
        label_a1 = Text("a₁ = 0.5", font_size=24, color=BLACK)
        label_a1.next_to(input_node1, DOWN, buff=0.2)
        group1 = VGroup(input_node1, label_a1)

        input_node2 = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        input_node2.move_to(LEFT * 3 + DOWN * 1)
        label_a2 = Text("a₂ = 0.4", font_size=24, color=BLACK)
        label_a2.next_to(input_node2, DOWN, buff=0.2)
        group2 = VGroup(input_node2, label_a2)

        # Create the output node.
        output_node = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        output_node.move_to(RIGHT * 1)
        label_n = Text("n", font_size=24, color=BLACK)
        label_n.move_to(output_node.get_center())
        group3 = VGroup(output_node, label_n)

        # Animate the creation of these nodes.
        self.play(Create(group1), Create(group2), Create(group3))
        self.wait(1)

        # Connect each input node to the output node.
        connection1 = Line(input_node1.get_center(), output_node.get_center(), stroke_width=2, color=BLACK)
        connection2 = Line(input_node2.get_center(), output_node.get_center(), stroke_width=2, color=BLACK)
        self.play(Create(connection1), Create(connection2))
        self.wait(1)

        # Specify weights for each connection.
        weight1 = 0.6
        weight2 = 0.9

        # Display weight labels along the connections.
        weight_label1 = Text(f"w₁ = {weight1}", font_size=24, color=BLACK)
        weight_label1.move_to((input_node1.get_center() + output_node.get_center()) / 2 + UP * 0.3)
        weight_label2 = Text(f"w₂ = {weight2}", font_size=24, color=BLACK)
        weight_label2.move_to((input_node2.get_center() + output_node.get_center()) / 2 + DOWN * 0.3)
        self.play(Write(weight_label1), Write(weight_label2))
        self.wait(1)

        # Show the multiplication at each connection.
        weighted1 = round(weight1 * 0.5, 4)  # 0.6*0.5 = 0.3
        weighted2 = round(weight2 * 0.4, 4)  # 0.9*0.4 = 0.36
        weighted_label1 = Text(f"{weight1}×0.5 = {weighted1}", font_size=24, color=BLACK)
        weighted_label1.next_to(connection1, UP, buff=0.1)
        weighted_label2 = Text(f"{weight2}×0.4 = {weighted2}", font_size=24, color=BLACK)
        weighted_label2.next_to(connection2, DOWN, buff=0.1)
        self.play(Write(weighted_label1), Write(weighted_label2))
        self.wait(1)

        # Sum the weighted inputs.
        total_weighted = round(weighted1 + weighted2, 4)  # 0.3+0.36 = 0.66
        sum_label = Text(f"Sum = {weighted1} + {weighted2} = {total_weighted}", font_size=24, color=BLACK)
        sum_label.to_edge(UP, buff=1)
        self.play(Write(sum_label))
        self.wait(1)

        # Pass the summed input through the activation function (ReLU).
        activation_value = round(max(0, total_weighted), 4)  # ReLU(0.66)=0.66
        activation_label = Text(f"ReLU({total_weighted}) = {activation_value}", font_size=24, color=BLACK)
        activation_label.to_edge(DOWN, buff=1)
        self.play(Write(activation_label))
        self.wait(1)

        # Highlight the activated output node by changing its color.
        activated_node = output_node.copy()
        activated_node.set_fill(GREEN, opacity=0.8)
        self.play(Transform(output_node, activated_node))
        self.wait(2)

        # Clean up the demonstration.
        self.play(
            FadeOut(VGroup(
                group1, group2, group3,
                connection1, connection2,
                weight_label1, weight_label2,
                weighted_label1, weighted_label2,
                sum_label, activation_label
            ))
        )

    def demonstrate_backpropagation(self):
        # Display a title for the backpropagation section.
        backprop_title = Text("Backpropagation", font_size=36, color=BLACK)
        backprop_title.to_edge(UP, buff=0.5)
        self.play(Write(backprop_title))
        self.wait(1)

        # Create nodes and reposition them away from the central/text area.
        # Nodes are now shifted to the lower left.
        input_node1 = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        input_node1.move_to(LEFT * 2 + DOWN * 1)
        label_a1 = Text("a₁ = 0.5", font_size=24, color=BLACK)
        label_a1.next_to(input_node1, DOWN, buff=0.2)
        group1 = VGroup(input_node1, label_a1)

        input_node2 = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        input_node2.move_to(LEFT * 2 + DOWN * 3)
        label_a2 = Text("a₂ = 0.4", font_size=24, color=BLACK)
        label_a2.next_to(input_node2, DOWN, buff=0.2)
        group2 = VGroup(input_node2, label_a2)

        output_node = Circle(radius=0.3, fill_color=ORANGE, fill_opacity=0.8)
        output_node.move_to(RIGHT * 2 + DOWN * 2)
        label_n = Text("n", font_size=24, color=BLACK)
        label_n.move_to(output_node.get_center())
        group3 = VGroup(output_node, label_n)

        self.play(Create(group1), Create(group2), Create(group3))
        self.wait(1)

        # Connect the nodes.
        connection1 = Line(input_node1.get_center(), output_node.get_center(), stroke_width=2, color=BLACK)
        connection2 = Line(input_node2.get_center(), output_node.get_center(), stroke_width=2, color=BLACK)
        self.play(Create(connection1), Create(connection2))
        self.wait(1)

        # Forward pass calculations.
        weight1 = 0.6
        weight2 = 0.9
        weighted1 = round(weight1 * 0.5, 4)  # 0.3
        weighted2 = round(weight2 * 0.4, 4)  # 0.36
        total_weighted = round(weighted1 + weighted2, 4)  # 0.66
        output_val = round(max(0, total_weighted), 4)  # 0.66

        forward_calc = Text(
            f"Forward Pass: {weight1}×0.5 + {weight2}×0.4 = {total_weighted}",
            font_size=24, color=BLACK
        )
        forward_calc.next_to(backprop_title, DOWN, buff=0.7)
        self.play(Write(forward_calc))
        self.wait(1)

        output_text = Text(
            f"Output (y) = ReLU({total_weighted}) = {output_val}",
            font_size=24, color=BLACK
        )
        output_text.next_to(forward_calc, DOWN, buff=0.3)
        self.play(Write(output_text))
        self.wait(1)

        target = 1.0
        target_text = Text(f"Target (t) = {target}", font_size=24, color=BLACK)
        target_text.next_to(output_text, DOWN, buff=0.3)
        self.play(Write(target_text))
        self.wait(1)

        error_val = round(output_val - target, 4)  # 0.66 - 1.0 = -0.34
        error_text = Text(
            f"Error: y - t = {output_val} - {target} = {error_val}",
            font_size=24, color=BLACK
        )
        error_text.next_to(target_text, DOWN, buff=0.3)
        self.play(Write(error_text))
        self.wait(1)

        grad_w1 = round(error_val * 0.5, 4)  # -0.17
        grad_w2 = round(error_val * 0.4, 4)  # -0.136
        grad_text = Text(
            f"Gradients: dL/dw₁ = {error_val}×0.5 = {grad_w1},  dL/dw₂ = {error_val}×0.4 = {grad_w2}",
            font_size=24, color=BLACK
        )
        grad_text.next_to(error_text, DOWN, buff=0.3)
        self.play(Write(grad_text))
        self.wait(1)

        alpha = 0.1
        new_w1 = round(weight1 - alpha * grad_w1, 4)  # 0.6 - 0.1*(-0.17) = 0.617
        new_w2 = round(weight2 - alpha * grad_w2, 4)  # 0.9 - 0.1*(-0.136)=0.9136
        update_text = Text(
            f"Updated Weights: w₁: {weight1} → {new_w1},  w₂: {weight2} → {new_w2}",
            font_size=24, color=BLACK
        )
        update_text.next_to(grad_text, DOWN, buff=0.3)
        self.play(Write(update_text))
        self.wait(1)

        # Fade out all backpropagation elements.
        self.play(
            FadeOut(VGroup(
                backprop_title, forward_calc, output_text, target_text,
                error_text, grad_text, update_text, group1, group2, group3,
                connection1, connection2
            ))
        )

    def create_subscript(self, base, sub, font_size=16):
        main_text = Text(base, font_size=font_size, color=BLACK)
        sub_text = Text(sub, font_size=font_size * 0.6, color=BLACK)
        sub_text.next_to(main_text, direction=DOWN + RIGHT, buff=0.05)
        return VGroup(main_text, sub_text)

    def create_neural_network(self, layer_sizes, node_radius=0.3):
        """Create a smaller neural network with the specified layer sizes."""
        network = VGroup()
        layers = []
        # Use smaller multipliers to scale down the network.
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            for j in range(size):
                node = Circle(radius=node_radius, fill_opacity=0.8)
                node.fill_color = ORANGE
                # Scale down the position factors.
                node.move_to(np.array([i * 2, (size - 1) * 0.6 - j * 1.2, 0]))
                if i == 0:
                    label = self.create_subscript("n", "i", font_size=16)
                elif i == len(layer_sizes) - 1:
                    label = self.create_subscript("n", "o", font_size=16)
                else:
                    label = self.create_subscript("n", f"{i}{j}", font_size=16)
                label.move_to(node.get_center())
                node_group = VGroup(node, label)
                layer.add(node_group)
            layers.append(layer)
            network.add(layer)
        # Connect nodes.
        for i in range(len(layers) - 1):
            for node1 in layers[i]:
                for node2 in layers[i + 1]:
                    conn = Line(
                        node1[0].get_center(),
                        node2[0].get_center(),
                        stroke_opacity=0.5,
                        stroke_width=1,
                        color=BLACK
                    )
                    network.add(conn)
        network.center()
        return network

    def highlight_network_flow(self, network):
        """Animate the flow of information through the network."""
        for node in network[0]:
            circle = node[0].copy()
            circle.set_fill(YELLOW, opacity=0.8)
            self.play(Transform(node[0], circle), run_time=0.5)
        for i in range(1, 4):
            for node in network[i]:
                circle = node[0].copy()
                circle.set_fill(YELLOW, opacity=0.8)
                self.play(Transform(node[0], circle), run_time=0.3)
        for layer in network[:4]:
            for node in layer:
                original_circle = Circle(
                    radius=node[0].radius,
                    fill_color=ORANGE,
                    fill_opacity=0.8
                ).move_to(node[0].get_center())
                self.play(Transform(node[0], original_circle), run_time=0.2)
