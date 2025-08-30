package relux

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"

	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
	"github.com/xDarkicex/relux/internal/serialize"
)

// Save serializes the trained network to an io.Writer using gob encoding.
// This enables persistent storage and network transfer of trained models.
func (n *Network) Save(w io.Writer) error {
	if n == nil {
		return fmt.Errorf("cannot save nil network")
	}
	if len(n.layers) == 0 {
		return fmt.Errorf("cannot save uninitialized network")
	}

	// Create snapshot using function closures (clean!)
	snapshot := serialize.CreateSnapshot(
		func() []layer.Layer { return n.layers },
		func() int { return n.inputSize },
		func() string { return n.loss.Name() },
	)

	encoder := gob.NewEncoder(w)
	if err := encoder.Encode(snapshot); err != nil {
		return fmt.Errorf("failed to encode network: %w", err)
	}
	return nil
}

// Load deserializes a network from an io.Reader using gob encoding.
// The network must be properly initialized before calling Load.
func (n *Network) Load(r io.Reader) error {
	if n == nil {
		return fmt.Errorf("cannot load into nil network")
	}

	var snapshot serialize.NetworkSnapshot
	decoder := gob.NewDecoder(r)
	if err := decoder.Decode(&snapshot); err != nil {
		return fmt.Errorf("failed to decode network: %w", err)
	}

	// Restore using function closures (clean!)
	serialize.RestoreNetwork(&snapshot,
		func(size int) { n.inputSize = size },
		func(l loss.Loss) { n.loss = l },
		func(layers []layer.Layer) { n.layers = layers },
	)

	return nil
}

// SaveFile saves the network to a file using gob encoding.
// Creates parent directories if they don't exist.
func (n *Network) SaveFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()

	if err := n.Save(file); err != nil {
		return fmt.Errorf("failed to save network to %s: %w", filename, err)
	}

	return nil
}

// LoadFile loads a network from a file using gob encoding.
func (n *Network) LoadFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", filename, err)
	}
	defer file.Close()

	if err := n.Load(file); err != nil {
		return fmt.Errorf("failed to load network from %s: %w", filename, err)
	}

	return nil
}

// LoadNetwork loads a pre-trained network from a file.
// Returns a new Network instance - no need to call NewNetwork first.
func LoadNetwork(filename string) (*Network, error) {
	net := &Network{} // Create empty network
	if err := net.LoadFile(filename); err != nil {
		return nil, err
	}
	return net, nil
}
