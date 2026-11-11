package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedFlatbuffersEnvelope is a serializable record used by flatbuffers.
type GeneratedFlatbuffersEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedFlatbuffers(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedFlatbuffersEnvelope{Name: "flatbuffers", Version: 1, Data: data})
}
func DecodeGeneratedFlatbuffers(raw []byte) ([]byte, error) {
	var e GeneratedFlatbuffersEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("flatbuffers: %w", err)
	}
	return e.Data, nil
}
