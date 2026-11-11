package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedTfliteEnvelope is a serializable record used by tflite.
type GeneratedTfliteEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedTflite(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedTfliteEnvelope{Name: "tflite", Version: 1, Data: data})
}
func DecodeGeneratedTflite(raw []byte) ([]byte, error) {
	var e GeneratedTfliteEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("tflite: %w", err)
	}
	return e.Data, nil
}
