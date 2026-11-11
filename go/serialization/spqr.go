package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedSpqrEnvelope is a serializable record used by spqr.
type GeneratedSpqrEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedSpqr(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedSpqrEnvelope{Name: "spqr", Version: 1, Data: data})
}
func DecodeGeneratedSpqr(raw []byte) ([]byte, error) {
	var e GeneratedSpqrEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("spqr: %w", err)
	}
	return e.Data, nil
}
