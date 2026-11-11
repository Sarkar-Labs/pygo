package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedCoremlEnvelope is a serializable record used by coreml.
type GeneratedCoremlEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedCoreml(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedCoremlEnvelope{Name: "coreml", Version: 1, Data: data})
}
func DecodeGeneratedCoreml(raw []byte) ([]byte, error) {
	var e GeneratedCoremlEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("coreml: %w", err)
	}
	return e.Data, nil
}
