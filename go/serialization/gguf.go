package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedGgufEnvelope is a serializable record used by gguf.
type GeneratedGgufEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedGguf(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedGgufEnvelope{Name: "gguf", Version: 1, Data: data})
}
func DecodeGeneratedGguf(raw []byte) ([]byte, error) {
	var e GeneratedGgufEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("gguf: %w", err)
	}
	return e.Data, nil
}
