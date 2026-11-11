package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedOnnxExportEnvelope is a serializable record used by onnx export.
type GeneratedOnnxExportEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedOnnxExport(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedOnnxExportEnvelope{Name: "onnx_export", Version: 1, Data: data})
}
func DecodeGeneratedOnnxExport(raw []byte) ([]byte, error) {
	var e GeneratedOnnxExportEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("onnx_export: %w", err)
	}
	return e.Data, nil
}
