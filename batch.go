package relux

import (
	"fmt"
	"sync"
)

// PredictBatch performs efficient bulk prediction on multiple inputs.
// Returns predictions in the same order as inputs.
func (n *Network) PredictBatch(X [][]float64) ([][]float64, error) {
	if n == nil || len(n.layers) == 0 {
		return nil, fmt.Errorf("network not initialized")
	}
	if len(X) == 0 {
		return [][]float64{}, nil
	}

	// Validate all inputs first
	for i, x := range X {
		if len(x) != n.inputSize {
			return nil, fmt.Errorf("input %d size %d does not match expected %d", i, len(x), n.inputSize)
		}
	}

	// Sequential processing for deterministic results
	results := make([][]float64, len(X))
	for i, x := range X {
		out := x
		for _, l := range n.layers {
			out = l.Forward(out)
		}
		results[i] = make([]float64, len(out))
		copy(results[i], out)
	}

	return results, nil
}

// PredictBatchConcurrent performs concurrent bulk prediction for high-throughput scenarios.
// Use when prediction latency is more important than deterministic ordering.
func (n *Network) PredictBatchConcurrent(X [][]float64, workers int) ([][]float64, error) {
	if n == nil || len(n.layers) == 0 {
		return nil, fmt.Errorf("network not initialized")
	}
	if len(X) == 0 {
		return [][]float64{}, nil
	}
	if workers <= 0 {
		workers = 4 // Default worker count
	}

	// Validate all inputs first
	for i, x := range X {
		if len(x) != n.inputSize {
			return nil, fmt.Errorf("input %d size %d does not match expected %d", i, len(x), n.inputSize)
		}
	}

	// Set up worker pool
	type job struct {
		index int
		input []float64
	}
	type result struct {
		index  int
		output []float64
		err    error
	}

	jobs := make(chan job, len(X))
	results := make(chan result, len(X))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				pred, err := n.Predict(j.input)
				results <- result{
					index:  j.index,
					output: pred,
					err:    err,
				}
			}
		}()
	}

	// Send jobs
	go func() {
		for i, x := range X {
			jobs <- job{index: i, input: x}
		}
		close(jobs)
	}()

	// Close results when workers finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	outputs := make([][]float64, len(X))
	for r := range results {
		if r.err != nil {
			return nil, fmt.Errorf("prediction failed for input %d: %w", r.index, r.err)
		}
		outputs[r.index] = r.output
	}

	return outputs, nil
}

// PredictSingle is an alias for Predict for API consistency.
func (n *Network) PredictSingle(x []float64) ([]float64, error) {
	return n.Predict(x)
}
