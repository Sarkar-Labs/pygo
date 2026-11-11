package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedMlirEnvelope is a serializable record used by mlir.
type GeneratedMlirEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedMlir(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedMlirEnvelope{Name: "mlir", Version: 1, Data: data})
}
func DecodeGeneratedMlir(raw []byte) ([]byte, error) {
	var e GeneratedMlirEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("mlir: %w", err)
	}
	return e.Data, nil
}
