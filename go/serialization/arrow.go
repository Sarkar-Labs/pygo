package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedArrowEnvelope is a serializable record used by arrow.
type GeneratedArrowEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedArrow(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedArrowEnvelope{Name: "arrow", Version: 1, Data: data})
}
func DecodeGeneratedArrow(raw []byte) ([]byte, error) {
	var e GeneratedArrowEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("arrow: %w", err)
	}
	return e.Data, nil
}
