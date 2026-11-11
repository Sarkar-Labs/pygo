package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedTensorrtEnvelope is a serializable record used by tensorrt.
type GeneratedTensorrtEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedTensorrt(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedTensorrtEnvelope{Name: "tensorrt", Version: 1, Data: data})
}
func DecodeGeneratedTensorrt(raw []byte) ([]byte, error) {
	var e GeneratedTensorrtEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("tensorrt: %w", err)
	}
	return e.Data, nil
}
