//go:build !rnxa
// +build !rnxa

package compute

import "fmt"

// Stubs for when rnxa is not available.
func newRnxaBackend() (ComputeBackend, error) {
	return nil, fmt.Errorf("rnxa backend not available (build with -tags rnxa)")
}

func newEnhancedRnxaBackend() (ComputeBackend, error) {
	return nil, fmt.Errorf("enhanced rnxa backend not available (build with -tags rnxa)")
}
