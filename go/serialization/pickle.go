package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedPickleEnvelope is a serializable record used by pickle.
type GeneratedPickleEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedPickle(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedPickleEnvelope{Name: "pickle", Version: 1, Data: data})
}
func DecodeGeneratedPickle(raw []byte) ([]byte, error) {
	var e GeneratedPickleEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("pickle: %w", err)
	}
	return e.Data, nil
}
