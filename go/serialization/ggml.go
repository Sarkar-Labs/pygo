package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedGgmlEnvelope is a serializable record used by ggml.
type GeneratedGgmlEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedGgml(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedGgmlEnvelope{Name: "ggml", Version: 1, Data: data})
}
func DecodeGeneratedGgml(raw []byte) ([]byte, error) {
	var e GeneratedGgmlEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("ggml: %w", err)
	}
	return e.Data, nil
}
