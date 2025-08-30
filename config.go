package relux

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
)

type InputSpec struct {
	Name string
	Size int
}

type LayerSpec struct {
	Units int
	Act   string // "relu", "sigmoid", "tanh", or ""
}

type Config struct {
	Inputs []InputSpec
	Hidden []LayerSpec
	Output LayerSpec
	Loss   string // "mse", "bce"
}

// Training options
type TrainOption func(*trainConfig)

type trainConfig struct {
	epochs       int
	batchSize    int
	learningRate float64
	shuffle      bool
	verbose      bool // Added verbose field
}

func Epochs(n int) TrainOption {
	return func(tc *trainConfig) { tc.epochs = n }
}

func BatchSize(n int) TrainOption {
	return func(tc *trainConfig) {
		if n <= 0 {
			n = 1 // Minimum batch size
		}
		tc.batchSize = n
	}
}

func LearningRate(lr float64) TrainOption {
	return func(tc *trainConfig) {
		if lr <= 0 || lr > 1.0 {
			// Could log warning or set to default
			// For now, clamp to reasonable range
			if lr <= 0 {
				lr = 0.001
			}
			if lr > 1.0 {
				lr = 1.0
			}
		}
		tc.learningRate = lr
	}
}

func Shuffle(s bool) TrainOption {
	return func(tc *trainConfig) { tc.shuffle = s }
}

// Added Verbose training option
func Verbose(v bool) TrainOption {
	return func(tc *trainConfig) { tc.verbose = v }
}

// Option is a functional option for configuring a Network.
type Option func(*builder) error

type builder struct {
	cfg Config
	rnd *rand.Rand
}

// NewNetwork constructs a multilayer perceptron with sensible defaults
// unless overridden via functional options.
func NewNetwork(opts ...Option) (*Network, error) {
	b := &builder{
		cfg: defaultConfig(),
		rnd: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	for _, opt := range opts {
		if err := opt(b); err != nil {
			return nil, err
		}
	}
	n := &Network{}
	if err := n.buildFrom(b); err != nil {
		return nil, err
	}
	return n, nil
}

func defaultConfig() Config {
	return Config{
		Inputs: []InputSpec{{Name: "x", Size: 2}},
		Hidden: []LayerSpec{
			{Units: 8, Act: "relu"},
		},
		Output: LayerSpec{Units: 1, Act: "sigmoid"},
		Loss:   "bce",
	}
}

// Enhanced error messages with context
func WithConfig(c Config) Option {
	return func(b *builder) error {
		if len(c.Inputs) == 0 {
			return fmt.Errorf("config validation failed: no inputs specified")
		}
		if c.Inputs[0].Size <= 0 {
			return fmt.Errorf("config validation failed: input size %d must be positive", c.Inputs[0].Size)
		}
		if c.Output.Units <= 0 {
			return fmt.Errorf("config validation failed: output units %d must be positive", c.Output.Units)
		}
		// Validate hidden layers
		for i, h := range c.Hidden {
			if h.Units <= 0 {
				return fmt.Errorf("config validation failed: hidden layer %d units %d must be positive", i, h.Units)
			}
		}
		b.cfg = c
		return nil
	}
}

// WithSeed sets a deterministic RNG for parameter initialization.
func WithSeed(seed int64) Option {
	return func(b *builder) error {
		b.rnd = rand.New(rand.NewSource(seed))
		return nil
	}
}

// toActivation resolves an activation name to an implementation.
func toActivation(name string) act.Activation {
	switch name {
	case "", "linear":
		return act.Identity()
	case "relu":
		return act.ReLU()
	case "sigmoid":
		return act.Sigmoid()
	case "tanh":
		return act.Tanh()
	default:
		return act.Identity()
	}
}

// toLoss resolves a loss name to an implementation.
func toLoss(name string) loss.Loss {
	switch name {
	case "mse":
		return loss.MSE()
	case "bce":
		return loss.BCE()
	default:
		return loss.MSE() // default
	}
}

// buildFrom wires layers based on the builder's config.
func (n *Network) buildFrom(b *builder) error {
	var layers []layer.Layer
	in := b.cfg.Inputs[0].Size

	// Hidden stack
	for _, h := range b.cfg.Hidden {
		layers = append(layers, layer.NewDense(in, h.Units, toActivation(h.Act), b.rnd))
		in = h.Units
	}

	// Output
	layers = append(layers, layer.NewDense(in, b.cfg.Output.Units, toActivation(b.cfg.Output.Act), b.rnd))

	n.layers = layers
	n.inputSize = b.cfg.Inputs[0].Size
	n.loss = toLoss(b.cfg.Loss)
	n.rnd = b.rnd

	return nil
}
