package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedLoadStateDictEnvelope is a serializable record used by load state dict.
type GeneratedLoadStateDictEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedLoadStateDict(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedLoadStateDictEnvelope{Name: "load_state_dict", Version: 1, Data: data})
}
func DecodeGeneratedLoadStateDict(raw []byte) ([]byte, error) {
	var e GeneratedLoadStateDictEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("load_state_dict: %w", err)
	}
	return e.Data, nil
}
