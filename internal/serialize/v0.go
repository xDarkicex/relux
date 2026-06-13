package serialize

import (
	"encoding/gob"
	"fmt"
	"io"
)

// V0Read reads a v0 (gob-encoded) .relux file. This is the
// existing format that has been the only relux format
// since the project's start; it's used for *Network (the
// MLP framework). The v1 path is a different file format
// with a "RELV" magic prefix; v0 has no magic.
//
// Callers typically don't call V0Read directly — they call
// Load() (in the relux package) which sniffs the magic and
// dispatches.
func V0Read(r io.Reader) (*NetworkSnapshot, error) {
	var snap NetworkSnapshot
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&snap); err != nil {
		return nil, fmt.Errorf("v0 (gob): decode: %w", err)
	}
	return &snap, nil
}
