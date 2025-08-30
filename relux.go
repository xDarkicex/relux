// Package relux provides a minimal, dependency-free multilayer perceptron (MLP)
// with a small, Go-idiomatic API focused on simplicity and reliability.
//
// Phase 5 implements advanced training features:
// - SGD with Momentum for faster convergence
// - Learning Rate Scheduling with exponential decay
// - Early Stopping to prevent overfitting
// - Enhanced Gradient Clipping for stability
// - Comprehensive training monitoring
//
// Example usage:
//
//	net, _ := relux.NewNetwork(relux.WithConfig(config))
//	net.Fit(X, Y,
//		relux.Epochs(5000),
//		relux.LearningRate(0.1),
//		relux.Momentum(0.9),
//		relux.LearningRateDecay(0.95, 500),
//		relux.EarlyStopping(100),
//		relux.Verbose(true),
//	)
package relux
