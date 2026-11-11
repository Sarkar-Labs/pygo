package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedQuantizeInt4Envelope is a serializable record used by quantize int4.
type GeneratedQuantizeInt4Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedQuantizeInt4(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedQuantizeInt4Envelope{Name: "quantize_int4", Version: 1, Data: data})
}
func DecodeGeneratedQuantizeInt4(raw []byte) ([]byte, error) {
	var e GeneratedQuantizeInt4Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("quantize_int4: %w", err)
	}
	return e.Data, nil
}
