package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedAwqEnvelope is a serializable record used by awq.
type GeneratedAwqEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedAwq(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedAwqEnvelope{Name: "awq", Version: 1, Data: data})
}
func DecodeGeneratedAwq(raw []byte) ([]byte, error) {
	var e GeneratedAwqEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("awq: %w", err)
	}
	return e.Data, nil
}
