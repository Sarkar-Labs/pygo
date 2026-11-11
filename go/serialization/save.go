package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedSaveEnvelope is a serializable record used by save.
type GeneratedSaveEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedSave(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedSaveEnvelope{Name: "save", Version: 1, Data: data})
}
func DecodeGeneratedSave(raw []byte) ([]byte, error) {
	var e GeneratedSaveEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("save: %w", err)
	}
	return e.Data, nil
}
