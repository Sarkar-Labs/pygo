package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedHloEnvelope is a serializable record used by hlo.
type GeneratedHloEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedHlo(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedHloEnvelope{Name: "hlo", Version: 1, Data: data})
}
func DecodeGeneratedHlo(raw []byte) ([]byte, error) {
	var e GeneratedHloEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("hlo: %w", err)
	}
	return e.Data, nil
}
