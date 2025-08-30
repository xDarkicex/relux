// Package relux provides a minimal, dependency-free multilayer perceptron (MLP)
// with a small, Go-idiomatic API focused on simplicity and reliability.
//
// Phase 2 implements full backpropagation training with configurable:
// - Network topology (input, hidden layers, output)
// - Activation functions (ReLU, Sigmoid, Tanh, Identity)
// - Loss functions (MSE, BCE)
// - Training options (epochs, learning rate, batch size)
//
// Example usage:
//
//	net, _ := relux.NewNetwork(
//	    relux.WithConfig(relux.Config{...}),
//	    relux.WithSeed(42),
//	)
//	net.Fit(X, Y, relux.Epochs(100), relux.LearningRate(0.01))
//	predictions, _ := net.Predict(x)
package relux
