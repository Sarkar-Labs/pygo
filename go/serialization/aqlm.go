package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedAqlmEnvelope is a serializable record used by aqlm.
type GeneratedAqlmEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedAqlm(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedAqlmEnvelope{Name: "aqlm", Version: 1, Data: data})
}
func DecodeGeneratedAqlm(raw []byte) ([]byte, error) {
	var e GeneratedAqlmEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("aqlm: %w", err)
	}
	return e.Data, nil
}
