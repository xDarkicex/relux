package compute

import (
	"fmt"
	"os"
	"strings"
)

// BackendOption configures backend selection
type BackendOption func(*backendConfig)

type backendConfig struct {
	backendType BackendType
	forceNative bool
}

// WithBackend specifies the backend type
func WithBackend(bt BackendType) BackendOption {
	return func(cfg *backendConfig) {
		cfg.backendType = bt
	}
}

// WithNativeOnly forces pure Go backend (for testing)
func WithNativeOnly() BackendOption {
	return func(cfg *backendConfig) {
		cfg.forceNative = true
	}
}

// NewComputeBackend creates the best available compute backend
func NewComputeBackend(opts ...BackendOption) ComputeBackend {
	cfg := &backendConfig{
		backendType: BackendAuto,
		forceNative: false,
	}

	for _, opt := range opts {
		opt(cfg)
	}

	// Check environment variable for override
	if env := os.Getenv("RELUX_BACKEND"); env != "" {
		switch strings.ToLower(env) {
		case "native", "cpu":
			cfg.forceNative = true
		case "rnxa":
			cfg.backendType = BackendRnxa
		case "auto":
			cfg.backendType = BackendAuto
		}
	}

	// Disable acceleration if requested
	if os.Getenv("RELUX_DISABLE_ACCELERATION") == "1" {
		cfg.forceNative = true
	}

	if cfg.forceNative {
		return newNativeBackend()
	}

	// Try backends in order of preference
	switch cfg.backendType {
	case BackendRnxa:
		if backend, err := tryRnxaBackend(); err == nil {
			return backend
		}
		// Fall back to native
		return newNativeBackend()

	case BackendAuto:
		// Auto-detect best backend
		// 1. Try rnxa (hardware acceleration)
		if backend, err := tryRnxaBackend(); err == nil {
			return backend
		}

		// 2. Fallback to native Go
		return newNativeBackend()

	default:
		return newNativeBackend()
	}
}

func tryRnxaBackend() (ComputeBackend, error) {
	backend, err := newRnxaBackend()
	if err != nil {
		return nil, fmt.Errorf("rnxa backend unavailable: %w", err)
	}

	if !backend.Available() {
		backend.Close()
		return nil, fmt.Errorf("rnxa backend not available")
	}

	return backend, nil
}

// GetAvailableBackends returns information about available backends
func GetAvailableBackends() []string {
	var backends []string

	// Native is always available
	backends = append(backends, "native (Pure Go)")

	// Try rnxa
	if backend, err := tryRnxaBackend(); err == nil {
		backends = append(backends, backend.Name())
		backend.Close()
	}

	return backends
}
