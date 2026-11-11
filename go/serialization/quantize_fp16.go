package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedQuantizeFp16Envelope is a serializable record used by quantize fp16.
type GeneratedQuantizeFp16Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedQuantizeFp16(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedQuantizeFp16Envelope{Name: "quantize_fp16", Version: 1, Data: data})
}
func DecodeGeneratedQuantizeFp16(raw []byte) ([]byte, error) {
	var e GeneratedQuantizeFp16Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("quantize_fp16: %w", err)
	}
	return e.Data, nil
}
