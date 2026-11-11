package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedOpenvinoEnvelope is a serializable record used by openvino.
type GeneratedOpenvinoEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedOpenvino(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedOpenvinoEnvelope{Name: "openvino", Version: 1, Data: data})
}
func DecodeGeneratedOpenvino(raw []byte) ([]byte, error) {
	var e GeneratedOpenvinoEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("openvino: %w", err)
	}
	return e.Data, nil
}
