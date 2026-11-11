package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedLoadEnvelope is a serializable record used by load.
type GeneratedLoadEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedLoad(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedLoadEnvelope{Name: "load", Version: 1, Data: data})
}
func DecodeGeneratedLoad(raw []byte) ([]byte, error) {
	var e GeneratedLoadEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("load: %w", err)
	}
	return e.Data, nil
}
