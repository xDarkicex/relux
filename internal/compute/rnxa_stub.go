//go:build !rnxa
// +build !rnxa

package compute

import "fmt"

// Stub implementation when rnxa is not available
func newRnxaBackend() (ComputeBackend, error) {
	return nil, fmt.Errorf("rnxa backend not available (build with -tags rnxa)")
}
