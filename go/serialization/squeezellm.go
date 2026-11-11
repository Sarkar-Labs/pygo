package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedSqueezellmEnvelope is a serializable record used by squeezellm.
type GeneratedSqueezellmEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedSqueezellm(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedSqueezellmEnvelope{Name: "squeezellm", Version: 1, Data: data})
}
func DecodeGeneratedSqueezellm(raw []byte) ([]byte, error) {
	var e GeneratedSqueezellmEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("squeezellm: %w", err)
	}
	return e.Data, nil
}
