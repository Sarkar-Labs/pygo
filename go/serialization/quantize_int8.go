package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedQuantizeInt8Envelope is a serializable record used by quantize int8.
type GeneratedQuantizeInt8Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedQuantizeInt8(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedQuantizeInt8Envelope{Name: "quantize_int8", Version: 1, Data: data})
}
func DecodeGeneratedQuantizeInt8(raw []byte) ([]byte, error) {
	var e GeneratedQuantizeInt8Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("quantize_int8: %w", err)
	}
	return e.Data, nil
}
