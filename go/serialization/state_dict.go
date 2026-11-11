package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedStateDictEnvelope is a serializable record used by state dict.
type GeneratedStateDictEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedStateDict(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedStateDictEnvelope{Name: "state_dict", Version: 1, Data: data})
}
func DecodeGeneratedStateDict(raw []byte) ([]byte, error) {
	var e GeneratedStateDictEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("state_dict: %w", err)
	}
	return e.Data, nil
}
