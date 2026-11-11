package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedGptqEnvelope is a serializable record used by gptq.
type GeneratedGptqEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedGptq(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedGptqEnvelope{Name: "gptq", Version: 1, Data: data})
}
func DecodeGeneratedGptq(raw []byte) ([]byte, error) {
	var e GeneratedGptqEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("gptq: %w", err)
	}
	return e.Data, nil
}
