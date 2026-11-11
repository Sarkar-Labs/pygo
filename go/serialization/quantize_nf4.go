package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedQuantizeNf4Envelope is a serializable record used by quantize nf4.
type GeneratedQuantizeNf4Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedQuantizeNf4(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedQuantizeNf4Envelope{Name: "quantize_nf4", Version: 1, Data: data})
}
func DecodeGeneratedQuantizeNf4(raw []byte) ([]byte, error) {
	var e GeneratedQuantizeNf4Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("quantize_nf4: %w", err)
	}
	return e.Data, nil
}
