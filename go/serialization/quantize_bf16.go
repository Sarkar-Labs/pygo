package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedQuantizeBf16Envelope is a serializable record used by quantize bf16.
type GeneratedQuantizeBf16Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedQuantizeBf16(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedQuantizeBf16Envelope{Name: "quantize_bf16", Version: 1, Data: data})
}
func DecodeGeneratedQuantizeBf16(raw []byte) ([]byte, error) {
	var e GeneratedQuantizeBf16Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("quantize_bf16: %w", err)
	}
	return e.Data, nil
}
