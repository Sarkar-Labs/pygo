package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedCheckpointEnvelope is a serializable record used by checkpoint.
type GeneratedCheckpointEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedCheckpoint(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedCheckpointEnvelope{Name: "checkpoint", Version: 1, Data: data})
}
func DecodeGeneratedCheckpoint(raw []byte) ([]byte, error) {
	var e GeneratedCheckpointEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("checkpoint: %w", err)
	}
	return e.Data, nil
}
